-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.Internal.TH
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Template haskell tools for finding Ops on NDArray and Symbol from dynamic library.
--
module MXNet.Core.Base.Internal.TH where

import Data.Char
import Data.List
import Data.Monoid
import Language.Haskell.TH

import MXNet.Core.NNVM.Base
import MXNet.Core.Base.Internal.Raw

-------------------------------------------------------------------------------

-- | Register NDArray ops.
registerNDArrayOps :: Bool      -- ^ If support "out" key in argument dictionary.
                   -> Q [Dec]
registerNDArrayOps mutable = runIO $ do
    names <- filter ((\c -> isLower c || c == '_') . head) <$> getNames
    concat <$> mapM (makeNDArrayFunc mutable) names

  where
    getNames :: IO [String]
    getNames = nnListAllOpNames >>= \(_, _, names) -> return names

-- | Prepend elements in the second map into the first one.
updateMap :: [(String, String)] -> [(String, String)] -> [(String, String)]
updateMap xs [] = xs
updateMap xs ((k, v) : ts) = case findIndex ((== k) . fst) xs of
                               Just _ -> xs `updateMap` ts
                               Nothing -> ((k, v) : xs) `updateMap` ts

-- | Generate the TH AST of a function for a NDArray op.
makeNDArrayFunc :: Bool     -- ^ If support "out" key in argument dictionary.
                -> String   -- ^ Function name.
                -> IO [Dec] -- ^ Generated signature and function definition.
makeNDArrayFunc mutable _name = do
    (_, handle) <- nnGetOpHandle _name
    (_, _ {- real name -}, desc, argc, argv, argtype, argdesc, _ {- kargs -}, rettype) <- mxSymbolGetAtomicSymbolInfo handle

    let notNdarrayFun = isUpper (head _name)

    let name = let str = if head _name == '_'
                            then _name
                            else toLower <$> _name
                in if mutable then str <> "'" else str

    let explicitArg = getExplicitArg argv argtype
        ndarrayArg = filter (\(v, t) -> t `startWith` "NDArray") explicitArg
        ordinaryArg = filter (\(v, t) -> not (t `startWith` "NDArray")) explicitArg
        implicitArg = getImplicitArg argv argtype
        hasImplicit = (not . null) implicitArg

    let forallArgT =
            if hasImplicit
               then ForallT [ KindedTV (mkName "kvs")
                                       (AppT ListT
                                             (AppT (ConT (mkName "KV")) StarT))
                            ]
                            [ AppT (ConT (mkName "ShowKV"))
                                   (VarT (mkName "kvs"))
                            , AppT (AppT (ConT (mkName "MatchKVList"))
                                         (VarT (mkName "kvs")))
                                   (makeKVListT implicitArg)
                            ]
               else id

        explicitArgT = (ConT . mkName . hsTypeName . snd) <$> explicitArg

        implicitArgT = if hasImplicit
                          then [AppT (ConT (mkName "HMap")) (VarT (mkName "kvs"))]
                          else undefined    -- will never be used

    let ndarrayArgP = (VarP . mkName . ("arg'" <>) . fst) <$> ndarrayArg
        ordinaryArgP = (VarP . mkName . ("arg'" <>) . fst) <$> ordinaryArg
        implicitArgP = if hasImplicit
                          then [VarP . mkName $ "varargs"]
                          else []
        returnArgP = if mutable
                        then [VarP (mkName "outputs")]
                        else []

    let ndargs = ListE $ (VarE . mkName . ("arg'" <>) . fst) <$> ndarrayArg
        dictargs = UInfixE (VarE (mkName "varArgK")) (VarE (mkName "zip")) (VarE (mkName "varArgV"))

    let func = NormalB . DoE $
            [ LetS [ ValD (VarP (mkName "defaultVarArgs"))
                          (NormalB $
                              foldr (\(name, t, value) acc -> AppE (AppE (AppE (VarE (mkName "add'"))
                                                                               (SigE (ConE (mkName "Proxy"))
                                                                                     (AppT (ConT (mkName "Proxy")) (LitT (StrTyLit name)))))
                                                                         (SigE (makeImplicitE t value) (ConT (mkName (hsTypeName t)))))
                                                                   acc)
                                    (VarE (mkName "empty"))
                                    implicitArg)
                          []
                   ]
            , LetS [ ValD (VarP (mkName "allArgs"))
                          (NormalB $
                              foldr (\(name, t) acc -> AppE (AppE (AppE (VarE (mkName "add'"))
                                                                        (SigE (ConE (mkName "Proxy"))
                                                                              (AppT (ConT (mkName "Proxy")) (LitT (StrTyLit name)))))
                                                                  (SigE (VarE (mkName ("arg'" <> name))) (ConT (mkName (hsTypeName t)))))
                                                         acc)
                                    (VarE (mkName $ if hasImplicit then "varargs" else "empty"))
                                    ordinaryArg
                          )
                          []
                   ]
            , LetS [ ValD (VarP (mkName "args"))
                          (NormalB $
                              AppE (VarE (mkName "dump"))
                                   (VarE (mkName "allArgs"))
                          )
                          []
                   , ValD (TupP [VarP (mkName "varArgK"), VarP (mkName "varArgV")])
                          (NormalB $
                              AppE (VarE (mkName "unzip"))
                                   (VarE (mkName "args"))
                          )
                          []
                   , ValD (VarP (mkName "outArg"))
                          (NormalB $
                              if mutable
                                 then AppE (ConE (mkName "Just")) (VarE (mkName "outputs"))
                                 else ConE (mkName "Nothing")
                          )
                          []
                   ]
            , BindS (TupP [VarP (mkName "_"), VarP (mkName "op")]) $
                AppE (VarE (mkName "nnGetOpHandle")) (LitE (StringL _name))
            , BindS (TupP [VarP (mkName "_"), VarP (mkName "res")]) $
                AppE (AppE (AppE (AppE (VarE (mkName "mxImperativeInvoke"))
                                       (VarE (mkName "op")))
                                 ndargs)
                           dictargs)
                     (VarE (mkName "outArg"))
            , NoBindS $
                AppE (VarE (mkName "return")) $
                    AppE (VarE (mkName "toResult")) (VarE (mkName "res"))
            ]

    let argT = explicitArgT <> (if hasImplicit then implicitArgT else [])
                            <> (if mutable then [AppT ListT (ConT (mkName "NDArrayHandle"))] else [])

        sig = SigD (mkName name) $
            ForallT [ PlainTV (mkName "r")]
                    [ AppT (ConT (mkName "NDArrayOpResult")) (VarT (mkName "r"))]
                    (forallArgT (foldr (\a b -> ArrowT `AppT` a `AppT` b)
                                       (AppT (ConT (mkName "IO")) (VarT (mkName "r")))
                                       argT))

        pragma = PragmaD $
            SpecialiseP (mkName name)
                        (forallArgT (foldr (\a b -> ArrowT `AppT` a `AppT` b)
                                    (AppT (ConT (mkName "IO")) (ConT (mkName "NDArrayHandle")))
                                    argT))
                        (Just Inline)
                        AllPhases

        fun = FunD (mkName name) [Clause (ndarrayArgP <> ordinaryArgP <> implicitArgP <> returnArgP) func []]


    return $ if argc == 0 || notNdarrayFun -- Is a function for symbol.
                          || _name `elem` [ "_NDArray", "_Native", "_arange"]
                          || explicitArg == []
                          || _name == "take" -- Operator @take@ will take two @SymbolHandle@ as arguments, can't be marshalled as strings.
                then []
                else [sig, fun, pragma]

  where

    -- | @startWith s t@ means @s@ starts with @t@.
    startWith :: String -> String -> Bool
    startWith s t = take (length t) s == t

    getExplicitArg :: [String]              -- ^ Argument names.
                   -> [String]              -- ^ Argument types.
                   -> [(String, String)]    -- ^ Return necessary arguments' name and type.
    getExplicitArg argv argtype = [t | Just t <- resolve <$> zip argv argtype]
        where
            resolve (v, t) = let ts = splitArgType t
                              in if "optional" `elem` ts
                                    then Nothing
                                    else Just (v, head ts)

    getImplicitArg :: [String]                      -- ^ Argument names.
                   -> [String]                      -- ^ Argument types.
                   -> [(String, String, String)]    -- ^ Return necessary arguments' names, types and default value.
    getImplicitArg argv argtype = [t | Just t <- resolve <$> zip argv argtype]
        where
            resolve (v, t) = let ts = splitArgType t
                              in  if "optional" `elem` ts
                                    then (\a -> (v, head ts, a)) <$> getDefault ts
                                    else Nothing

            getDefault = stripPrefix "default=" . head . filter (isPrefixOf "default=")

    makeImplicitE :: String -> String -> Exp
    makeImplicitE t value = case t of
                                 "boolean" -> ConE (mkName value)
                                 "float" -> LitE (RationalL (toRational (read value :: Float)))
                                 "double" -> LitE (RationalL (toRational (read value :: Double)))
                                 "real_t" -> LitE (RationalL (toRational (read value :: Float)))
                                 'i':'n':'t':_ -> LitE (IntegerL (read (tail $ init value) :: Integer))
                                 'l':'o':'n':'g':_ -> LitE (IntegerL (read (tail $ init value) :: Integer))
                                 '{':_ -> LitE (StringL (tail $ init value))
                                 "Shape(tuple)" -> LitE (StringL value)
                                 s -> VarE (mkName ("unknown implicit argument type: " <> s <> " with value: " <> value))

    splitArgType :: String -> [String]
    splitArgType (' ' : xs) = splitArgType xs
    splitArgType ts = case break (== ',') ts of
                       ([], _) -> []
                       (t, []) -> [t]
                       (t, _:xs) -> t : splitArgType xs

    makeKVListT :: [(String, String, String)] -> Type
    makeKVListT args = foldr combineKV PromotedNilT ((\(v, t, _) -> makeKV v t) <$> args)
        where
            makeKV v t = AppT (AppT (PromotedT (mkName ":="))
                                    (LitT (StrTyLit v)))
                              (ConT (mkName (hsTypeName t)))
            combineKV a acc = AppT (AppT (PromotedT (mkName ":")) a) acc

-- | Translate mxnet's type name to Haskell's type name.
hsTypeName :: String -> String
hsTypeName s = case s of
                 "boolean" -> "Bool"
                 "float" -> "Float"
                 "double" -> "Double"
                 "real_t" -> "Float"
                 'i':'n':'t':_ -> "Int"
                 'l':'o':'n':'g':_ -> "Int"
                 "string" -> "String"
                 "NDArray" -> "NDArrayHandle"
                 "Symbol" -> "SymbolHandle"
                 "NDArray[]" -> "[NDArray]"
                 "Symbol[]" -> "[SymbolHandle]"
                 '{':_ -> "String"
                 "Shape(tuple)" -> "String"
                 s -> "unknown type name: " <> s --- error $ "Unknown type name or optional argument: " <> s

viewFunc :: Bool -> String -> IO ()
viewFunc mutable name = makeNDArrayFunc mutable name >>= putStrLn . pprint


-------------------------------------------------------------------------------

registerSymbolOps :: Q [Dec]
registerSymbolOps = return []

-------------------------------------------------------------------------------

