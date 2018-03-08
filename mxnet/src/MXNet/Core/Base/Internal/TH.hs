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

import           Data.Char
import           Data.List
import           Data.Monoid
import           Language.Haskell.TH

import           MXNet.Core.NNVM.Internal
import           MXNet.Core.Base.Internal

-------------------------------------------------------------------------------

-- | Register NDArray ops.
registerNDArrayOps :: Bool      -- ^ If support "out" key in argument dictionary.
                   -> Q [Dec]
registerNDArrayOps mutable = runIO $ do
    names <- mxListAllOpNames
    concat <$> mapM (register mutable) names
  where
    register mutable _name = do
        (_, handle) <- nnGetOpHandle _name
        (_, desc, _, argv, argtype, _, _, _) <- mxSymbolGetAtomicSymbolInfo handle
        makeNDArrayFunc mutable _name desc argv argtype

-- | Register symbol functions.
registerSymbolOps :: Q [Dec]
registerSymbolOps = runIO $ do
    names <- mxListAllOpNames
    concat <$> mapM register names
  where
    register _name = do
        (_, handle) <- nnGetOpHandle _name
        (_, desc, _, argv, argtype, _, _, _) <- mxSymbolGetAtomicSymbolInfo handle
        makeSymbolFunc _name desc argv argtype

-------------------------------------------------------------------------------
-- | Generate the TH AST of a function for a NDArray op.
makeNDArrayFunc :: Bool     -- ^ If support "out" key in argument dictionary.
                -> String   -- ^ Function's name.
                -> String   -- ^ Function's description.
                -> [String] -- ^ Function's argument names.
                -> [String] -- ^ Function's argument types.
                -> IO [Dec] -- ^ Generated signature and function definition.
makeNDArrayFunc mutable _name desc argv argtype = do

    let deprecated = desc `startWith` "DEPRECATED" ||
                     _name == "Softmax" -- Softmax is renamed to SoftmaxOutput

    let alias = _name `elem` ["Concat", "Pad", "Flatten", "Reshape"]

    let name = let str = if head _name == '_'
                            then _name
                            else if _name == "where"
                               then "where_"
                               else toLower <$> _name
                in if mutable then str <> "'" else str

    let explicitArg = getExplicitArg argv argtype
        ndarrayArg = filter (\(_, t) -> t `startWith` "NDArray" || t `startWith` "Symbol") explicitArg
        ordinaryArg = filter (\(_, t) -> not (t `startWith` "NDArray" || t `startWith` "Symbol")) explicitArg
        implicitArg = getImplicitArg argv argtype
        hasImplicit = (not . null) implicitArg

    let forallArgT = makeForallArgT implicitArg

        explicitArgT = (makeHsType . snd) <$> explicitArg

        implicitArgT = if hasImplicit
                          then [AppT (ConT (mkName "HMap")) (VarT (mkName "kvs"))]
                          else error "Impossible: no implicit available."    -- will never be evaluated

    let ndarrayArgP = (VarP . mkName . ("arg'" <>) . fst) <$> ndarrayArg
        ordinaryArgP = (VarP . mkName . ("arg'" <>) . fst) <$> ordinaryArg
        implicitArgP = if hasImplicit
                          then [VarP . mkName $ "varargs"]
                          else []
        returnArgP = if mutable
                        then [VarP (mkName "outputs")]
                        else []

    let ndargs = foldr (\(v, t) args-> case makeHsType t of
                                             ConT _ -> UInfixE (VarE . mkName . ("arg'" <>) $ v) (ConE . mkName $ ":") args
                                             AppT ListT _ -> UInfixE (VarE . mkName . ("arg'" <>) $ v) (VarE . mkName $ "++") args
                                             _ -> error "Impossible: not a valid haskell type representation.")
                        (ListE [])
                        ndarrayArg

        dictargs = UInfixE (VarE (mkName "varArgK")) (VarE (mkName "zip")) (VarE (mkName "varArgV"))

    let func = NormalB . DoE $
            [ LetS [ ValD (VarP (mkName "allArgs"))
                          (NormalB $
                              foldr (\(name, t) acc -> AppE (AppE (AppE (VarE (mkName "add'"))
                                                                        (SigE (ConE (mkName "Proxy"))
                                                                              (AppT (ConT (mkName "Proxy")) (LitT (StrTyLit name)))))
                                                                  (SigE (VarE (mkName ("arg'" <> name))) (makeHsType t)))
                                                         acc)
                                    (VarE (mkName $ if hasImplicit then "varargs" else "nil"))
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
            , BindS (VarP (mkName "res")) $
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


    return $ if null argv || deprecated
                          || alias
                          || _name `elem` ["_NDArray", "_Native", "_arange"]
                          || _name `elem` ["cast", "crop"]  -- duplicate with "Cast" and "Crop"
                          -- || null explicitArg
                          || _name == "take" -- Operator @take@ will take two @SymbolHandle@ as arguments, can't be marshalled as strings.
                then []
                else [sig, fun, pragma]
  where
    -- | Translate mxnet's type name to Haskell's type name.
    makeHsType :: String -> Type
    makeHsType s = case s of
                    "boolean" -> ConT . mkName $ "Bool"
                    "float" -> ConT . mkName $ "Float"
                    "double" -> ConT . mkName $ "Double"
                    "real_t" -> ConT . mkName $ "Float"
                    'i':'n':'t':_ -> ConT . mkName $ "Int"
                    'l':'o':'n':'g':_ -> ConT . mkName $ "Int"
                    "string" -> ConT . mkName $ "String"
                    "NDArray" -> ConT . mkName $ "NDArrayHandle"
                    "NDArray-or-Symbol" -> ConT . mkName $ "NDArrayHandle"
                    "NDArray-or-Symbol[]" -> AppT ListT . ConT . mkName $ "NDArrayHandle"
                    "Symbol" -> ConT . mkName $ "NDArrayHandle"
                    "NDArray[]" -> AppT ListT . ConT . mkName $ "NDArrayHandle"
                    "Symbol[]" -> AppT ListT . ConT . mkName $ "NDArrayHandle"
                    "Symbol or Symbol[]" -> AppT ListT . ConT . mkName $ "NDArrayHandle"
                    '{':_ -> ConT . mkName $ "String"
                    "Shape(tuple)" -> ConT . mkName $ "String"
                    "tuple of <float>"  -> AppT ListT . ConT . mkName $ "Float"
                    "tuple of <double>" -> AppT ListT . ConT . mkName $ "Double"
                    s -> ConT . mkName $ "unknown type name: " <> s

    -- | Generate type signatures for implicit arguments.
    makeKVListT :: [(String, String, String)]   -- ^ [(name, type, default value)]
                -> Type
    makeKVListT args = foldr combineKV PromotedNilT ((\(v, t, _) -> makeKV v t) <$> args)
        where
            makeKV v t = AppT (AppT (PromotedT (mkName ":="))
                                    (LitT (StrTyLit v)))
                              (makeHsType t)
            combineKV a acc = AppT (AppT (PromotedT (mkName ":")) a) acc

    -- | Make forall arguments signature according to it's implicit argument.
    makeForallArgT :: [(String, String, String)]    -- ^ Implicit arguments, (name, type, default value)
                   -> (Type -> Type)
    makeForallArgT [] = id
    makeForallArgT implicitArg =
        ForallT [ KindedTV (mkName "kvs")
                           (AppT ListT
                                 (AppT (ConT (mkName "KV")) StarT))
                ]
                [ AppT (ConT (mkName "ShowKV"))
                       (VarT (mkName "kvs"))
                , AppT (AppT (ConT (mkName "MatchKVList"))
                             (VarT (mkName "kvs")))
                       (makeKVListT implicitArg)
                ]

-- | Generate the TH AST of a function for a Symbol op.
makeSymbolFunc :: String   -- ^ Function's name.
               -> String   -- ^ Function's description.
               -> [String] -- ^ Function's argument names.
               -> [String] -- ^ Function's argument types.
               -> IO [Dec] -- ^ Generated signature and function definition.
makeSymbolFunc _name desc argv argtype = do

    let deprecated = desc `startWith` "DEPRECATED" ||
                     _name == "Softmax" -- Softmax is renamed to SoftmaxOutput

    let alias = _name `elem` ["Concat", "Pad", "Flatten", "Reshape"]

    let name = let str = if head _name == '_'
                            then _name
                            else if _name == "where"
                               then "where_"
                               else toLower <$> _name
                in str

    let explicitArg = getExplicitArg argv argtype
        ndarrayArg = filter (\(v, t) -> t `startWith` "NDArray" || t `startWith` "Symbol") explicitArg
        ordinaryArg = filter (\(v, t) -> not (t `startWith` "NDArray" || t `startWith` "Symbol")) explicitArg
        implicitArg = getImplicitArg argv argtype
        hasImplicit = (not . null) implicitArg

    let forallArgT = makeForallArgT implicitArg

        explicitArgT = (makeHsType . snd) <$> explicitArg

        implicitArgT = if hasImplicit
                          then [AppT (ConT (mkName "HMap")) (VarT (mkName "kvs"))]
                          else error "Impossible: no implicit available."    -- will never be evaluated

    let nameArgP = [VarP . mkName $ "name"]
        ndarrayArgP = (VarP . mkName . ("arg'" <>) . fst) <$> ndarrayArg
        ordinaryArgP = (VarP . mkName . ("arg'" <>) . fst) <$> ordinaryArg
        implicitArgP = if hasImplicit
                          then [VarP . mkName $ "varargs"]
                          else []

    let ndargs = foldr (\(v, t) args -> case makeHsType t of
                                             ConT _ -> UInfixE (VarE . mkName . ("arg'" <>) $ v) (ConE . mkName $ ":") args
                                             AppT ListT _ -> UInfixE (VarE . mkName . ("arg'" <>) $ v) (VarE . mkName $ "++") args
                                             _ -> error "Impossible: not a valid haskell type representation.")
                        (ListE [])
                        ndarrayArg

    let func = NormalB . DoE $
            [ LetS [ ValD (VarP (mkName "allArgs"))
                          (NormalB $
                              foldr (\(name, t) acc -> AppE (AppE (AppE (VarE (mkName "add'"))
                                                                        (SigE (ConE (mkName "Proxy"))
                                                                              (AppT (ConT (mkName "Proxy")) (LitT (StrTyLit name)))))
                                                                  (SigE (VarE (mkName ("arg'" <> name))) (makeHsType t)))
                                                         acc)
                                    (VarE (mkName $ if hasImplicit then "varargs" else "nil"))
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
                   ]
            , BindS (TupP [VarP (mkName "_"), VarP (mkName "op")]) $
                AppE (VarE (mkName "nnGetOpHandle")) (LitE (StringL _name))
            , LetS [ ValD (VarP (mkName "nargs"))
                          (NormalB (AppE (VarE (mkName "fromIntegral"))
                                         (AppE (VarE (mkName "length"))
                                               (VarE (mkName "varArgK")))))
                          []
                   ]
            , BindS (TupP [VarP (mkName "_"), VarP (mkName "sym")]) $
                AppE (AppE (AppE (AppE (VarE (mkName "mxSymbolCreateAtomicSymbol"))
                                       (VarE (mkName "op")))
                                 (VarE (mkName "nargs")))
                           (VarE (mkName "varArgK")))
                     (VarE (mkName "varArgV"))
            , BindS (VarP (mkName "_")) $
                AppE (AppE (AppE (AppE (VarE (mkName "nnSymbolCompose"))
                                       (VarE (mkName "sym")))
                                 (VarE (mkName "name")))
                           (ListE []))
                     ndargs
            , NoBindS $
                AppE (VarE (mkName "return"))
                     (VarE (mkName "sym"))
            ]

    let argT = (ConT . mkName $ "String") : explicitArgT <> (if hasImplicit then implicitArgT else [])

        sig = SigD (mkName name) $
            forallArgT (foldr (\a b -> ArrowT `AppT` a `AppT` b)
                       (AppT (ConT (mkName "IO")) (ConT (mkName "SymbolHandle")))
                       argT)

        fun = FunD (mkName name) [Clause (nameArgP <> ndarrayArgP <> ordinaryArgP <> implicitArgP) func []]


    return $ if null argv || deprecated
                          || alias
                          || _name `elem` ["_NDArray", "_Native", "_arange"]
                          || _name `elem` ["cast", "crop"]  -- duplicate with "Cast" and "Crop"
                          -- || null explicitArg
                          || _name == "take" -- Operator @take@ will take two @SymbolHandle@ as arguments, can't be marshalled as strings.
                          || _name == "where"
                then []
                else [sig, fun]
  where
    -- | Translate mxnet's type name to Haskell's type name.
    makeHsType :: String -> Type
    makeHsType s = case s of
                    "boolean" -> ConT . mkName $ "Bool"
                    "float" -> ConT . mkName $ "Float"
                    "double" -> ConT . mkName $ "Double"
                    "real_t" -> ConT . mkName $ "Float"
                    'i':'n':'t':_ -> ConT . mkName $ "Int"
                    'l':'o':'n':'g':_ -> ConT . mkName $ "Int"
                    "string" -> ConT . mkName $ "String"
                    "NDArray" -> ConT . mkName $ "SymbolHandle"
                    "Symbol" -> ConT . mkName $ "SymbolHandle"
                    "NDArray-or-Symbol" -> ConT . mkName $ "SymbolHandle"
                    "NDArray-or-Symbol[]" -> AppT ListT . ConT . mkName $ "SymbolHandle"
                    "NDArray[]" -> AppT ListT . ConT . mkName $ "SymbolHandle"
                    "Symbol[]" -> AppT ListT . ConT . mkName $ "SymbolHandle"
                    "Symbol or Symbol[]" -> AppT ListT . ConT . mkName $ "SymbolHandle"
                    '{':_ -> ConT . mkName $ "String"
                    "Shape(tuple)" -> ConT . mkName $ "String"
                    "tuple of <float>" -> AppT ListT . ConT . mkName $ "Float"
                    "tuple of <double>" -> AppT ListT . ConT . mkName $ "Double"
                    s -> ConT . mkName $ "unknown type name: " <> s

    -- | Generate type signatures for implicit arguments.
    makeKVListT :: [(String, String, String)]   -- ^ [(name, type, default value)]
                -> Type
    makeKVListT args = foldr combineKV PromotedNilT ((\(v, t, _) -> makeKV v t) <$> args)
        where
            makeKV v t = AppT (AppT (PromotedT (mkName ":="))
                                    (LitT (StrTyLit v)))
                              (makeHsType t)
            combineKV a acc = AppT (AppT (PromotedT (mkName ":")) a) acc

    -- | Make forall arguments signature according to it's implicit argument.
    makeForallArgT :: [(String, String, String)]    -- ^ Implicit arguments, (name, type, default value)
                   -> (Type -> Type)
    makeForallArgT [] = id
    makeForallArgT implicitArg =
        ForallT [ KindedTV (mkName "kvs")
                           (AppT ListT
                                 (AppT (ConT (mkName "KV")) StarT))
                ]
                [ AppT (ConT (mkName "ShowKV"))
                       (VarT (mkName "kvs"))
                , AppT (AppT (ConT (mkName "MatchKVList"))
                             (VarT (mkName "kvs")))
                       (makeKVListT implicitArg)
                ]

-------------------------------------------------------------------------------

-- | @startWith s t@ means @s@ starts with @t@.
startWith :: String -> String -> Bool
startWith s t = take (length t) s == t

-- | Prepend elements in the second map into the first one.
updateMap :: [(String, String)] -> [(String, String)] -> [(String, String)]
updateMap xs [] = xs
updateMap xs ((k, v) : ts) = case findIndex ((== k) . fst) xs of
                               Just _ -> xs `updateMap` ts
                               Nothing -> ((k, v) : xs) `updateMap` ts

-- | Split argument string with ",", split name, type, default value and required information.
splitArgType :: String -> [String]
splitArgType (' ' : xs) = splitArgType xs
splitArgType ts = case break (== ',') ts of
                    ([], _) -> []
                    (t, []) -> [t]
                    (t, _:xs) -> t : splitArgType xs

-- | Get explicit arguments from all arugments.
getExplicitArg :: [String]              -- ^ Argument names.
               -> [String]              -- ^ Argument types.
               -> [(String, String)]    -- ^ Return necessary arguments' name and type.
getExplicitArg argv argtype = [t | Just t <- resolve <$> zip argv argtype]
    where
        resolve (v, t) = let ts = splitArgType t
                          in if "optional" `elem` ts
                                then Nothing
                                else if null ts                            -- Seems that `tuple of <float>` can't be exported correctly by mxSymbolGetAtomicSymbolInfo.
                                        then Just (v, "tuple of <float>")
                                        else Just (v, head ts)

-- | Get implicit arguments from all arguments.
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
