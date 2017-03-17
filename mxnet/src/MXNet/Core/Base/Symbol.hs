-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.Symbol
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Symbol module.
--
{-# LANGUAGE RecordWildCards #-}

module MXNet.Core.Base.Symbol where

import           Control.Exception
import           Control.Monad
import           Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM
import           Data.Monoid
import           Foreign.Ptr (nullPtr)
import           System.IO.Unsafe
import           Unsafe.Coerce (unsafeCoerce)

import           MXNet.Core.Base.DType
import           MXNet.Core.Base.Internal
import qualified MXNet.Core.Base.Internal.TH.Symbol as I
import           MXNet.Core.Base.HMap
import           MXNet.Core.Base.Executor hiding (getHandle)
import           MXNet.Core.Base.NDArray hiding (getHandle)
import qualified MXNet.Core.Base.NDArray as NDArray (getHandle)
import           MXNet.Core.NNVM.Internal

-- | Type alias for variable.
newtype Symbol a = Symbol { getHandle :: SymbolHandle }

instance DType a => Show (Symbol a) where
    show sym = unsafePerformIO $ do
        (_, str) <- mxSymbolPrint (getHandle sym)
        return str

-- | Make a new symbolic variable with given name.
variable :: DType a
        => String           -- ^ Name.
        -> IO (Symbol a)    -- ^ Result variable.
variable name = do
    (_, handle) <- mxSymbolCreateVariable name
    return $ Symbol handle

-- | Get the name of a given variable.
getName :: DType a => Symbol a -> IO String
getName = mxSymbolGetName . getHandle >=> \(_, nm, _) -> return nm

-- | Get specified attribute of symbol.
getAttr :: DType a => Symbol a -> String -> IO (Maybe String)
getAttr sym key = do
    (_, s, success) <- mxSymbolGetAttr (getHandle sym) key
    return $ if success == 0    -- 0 when success, -1 when failure happens
                then Just s
                else Nothing

-- | Set specified attribute of symbol.
setAttr :: DType a => Symbol a -> String -> String -> IO ()
setAttr sym key value = void $ mxSymbolSetAttr (getHandle sym) key value

-- | Infer the shape of the given symbol, return the in, out and auxiliary shape size.
infershape :: DType a => Symbol a -> [String] -> IO ([[Int]], [[Int]], [[Int]])
infershape sym args = do
    (_, arg, out, aux) <- mxSymbolInferShape (getHandle sym) args [0] []
    return (arg, out, aux)

-- | Get the autodiff of current symbol.
-- This function can only be used if current symbol is a loss function.
grad :: DType a => Symbol a -> [String] -> IO (Symbol a)
grad sym args = do
    let nargs = fromIntegral (length args)
    (_, handle) <- mxSymbolGrad (getHandle sym) nargs args
    return $ Symbol handle

-- | List all input arguments.
listInputs :: DType a => Symbol a -> IO [String]
listInputs sym = snd <$> mxSymbolListArguments (getHandle sym)

-- | List all output results.
listOutputs :: DType a => Symbol a -> IO [String]
listOutputs sym = snd <$> mxSymbolListOutputs (getHandle sym)

-- | List all auxiliary states.
listAuxiliaries :: DType a => Symbol a -> IO [String]
listAuxiliaries sym = snd <$> mxSymbolListAuxiliaryStates (getHandle sym)

-- | Bind with explicit argument mapping (name -- value mapping).
bind :: DType a
     => Symbol a
     -> Context
     -> HashMap String (NDArray a)
     -> IO (Executor a)
bind sym Context{..} args = do
    inputs <- genNDArrayMapping args <$> listInputs sym
    -- req_map = {'null': 0, 'write': 1, 'add': 3}
    let req_types = replicate (HM.size inputs) 1        -- use default value.
    (_, exec) <- mxExecutorBind (getHandle sym)
                                deviceType
                                deviceId
                                (fromIntegral (HM.size inputs))     -- length of input arguments.
                                (NDArray.getHandle <$> HM.elems inputs)
                                (replicate (HM.size inputs) (unsafeCoerce nullPtr))
                                req_types
                                0                                   -- length of auxiliary states.
                                []                                  -- no auxiliary states.
    return $ Executor exec
  where
    -- | Get ndarray lists handles from input arguments.
    genNDArrayMapping args arg_names = HM.fromList (genfn <$> arg_names)
      where
        genfn nm = case HM.lookup nm args of
                        Just v -> (nm, v)
                        Nothing -> throw . userError $ "getNDArrayInputs: no argument " <> nm

-- | Bind without explicit argument mapping (name -- value mapping).
bind' :: DType a
      => Symbol a
      -> Context
      -> [NDArray a]
      -> IO (Executor a)
bind' sym Context{..} args = do
    inputs <- genNDArrayMapping args <$> listInputs sym
    -- req_map = {'null': 0, 'write': 1, 'add': 3}
    let req_types = replicate (HM.size inputs) 1        -- use default value.
    (_, exec) <- mxExecutorBind (getHandle sym)
                                deviceType
                                deviceId
                                (fromIntegral (HM.size inputs))     -- length of input arguments.
                                (NDArray.getHandle <$> HM.elems inputs)
                                (replicate (HM.size inputs) (unsafeCoerce nullPtr))
                                req_types
                                0                                   -- length of auxiliary states.
                                []                                  -- no auxiliary states.
    return $ Executor exec
  where
    -- | Get ndarray lists handles from input arguments without explicit argument names.
    genNDArrayMapping args names =
        assert (length args == length names) $
            HM.fromList (zip names args)

instance DType a => Num (Symbol a) where
    (+) sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Plus (name1 <> "+" <> name2) handle1 handle2
    (-) sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Minus (name1 <> "-" <> name2) handle1 handle2
    (*) sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Mul (name1 <> "*" <> name2) handle1 handle2
    abs sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.abs ("|" <> name1 <> "|") handle1
    negate sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.negative ("(-" <> name1 <> ")") handle1
    signum = error "Unsupported operation: signum(Symbol)"
    fromInteger = error "Unsupported operation: fromInteger(Symbol)"

instance DType a => Fractional (Symbol a) where
    (/) sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Div (name1 <> "/" <> name2) handle1 handle2
    fromRational = error "Unsupported operation: fromRational(Symbol)"

instance DType a => Floating (Symbol a) where
    exp sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.exp ("exp(" <> name1 <> ")") handle1
    log sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.log ("log(" <> name1 <> ")") handle1
    sqrt sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.sqrt ("sqrt(" <> name1 <> ")") handle1
    sin sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.sin ("sin(" <> name1 <> ")") handle1
    cos sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.cos ("cos(" <> name1 <> ")") handle1
    tan sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.tan ("tan(" <> name1 <> ")") handle1
    sinh sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.sinh ("sinh(" <> name1 <> ")") handle1
    cosh sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.cosh ("cosh(" <> name1 <> ")") handle1
    tanh sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- getName sym1
        I.tanh ("tanh(" <> name1 <> ")") handle1

instance Tensor Symbol where
    (.+) sym value = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._PlusScalar (name1 <> "+" <> show value) handle (realToFrac value)
    {-# INLINE (.+) #-}
    (.-) sym value = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._MinusScalar (name1 <> "-" <> show value) handle (realToFrac value)
    {-# INLINE (.-) #-}
    (.*) sym value = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._MulScalar (name1 <> "*" <> show value) handle (realToFrac value)
    {-# INLINE (.*) #-}
    (./) sym value = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._DivScalar (name1 <> "/" <> show value) handle (realToFrac value)
    {-# INLINE (./) #-}
    (.^) sym value = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._PowerScalar (name1 <> "^" <> show value) handle (realToFrac value)
    {-# INLINE (.^) #-}
    (..-) value sym = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._RMinusScalar (show value <> "-" <> name1) handle (realToFrac value)
    {-# INLINE (..-) #-}
    (../) value sym = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._RDivScalar (show value <> "/" <> name1) handle (realToFrac value)
    {-# INLINE (../) #-}
    (..^) value sym = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._RPowerScalar (show value <> "^" <> name1) handle (realToFrac value)
    {-# INLINE (..^) #-}

    maximum sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Maximum ("maximum(" <> name1 <> "," <> name2 <> ")") handle1 handle2
    {-# INLINE maximum #-}
    maximum' sym scalar = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._MaximumScalar ("maximum'(" <> name1 <> "," <> show scalar <> ")") handle (realToFrac scalar)
    {-# INLINE maximum' #-}
    minimum sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Minimum ("minimum(" <> name1 <> "," <> name2 <> ")") handle1 handle2
    {-# INLINE minimum #-}
    minimum' sym scalar = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._MinimumScalar ("minimum'(" <> name1 <> "," <> show scalar <> ")") handle (realToFrac scalar)
    {-# INLINE minimum' #-}
    equal sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_equal (name1 <> "==" <> name2) handle1 handle2
    {-# INLINE equal #-}
    equal' sym scalar = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._equal_scalar (name1 <> "==" <> show scalar) handle (realToFrac scalar)
    {-# INLINE equal' #-}
    notEqual sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_not_equal (name1 <> "/=" <> name2) handle1 handle2
    {-# INLINE notEqual #-}
    notEqual' sym scalar = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._not_equal_scalar (name1 <> "/=" <> show scalar) handle (realToFrac scalar)
    {-# INLINE notEqual' #-}
    greater sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_greater (name1 <> ">" <> name2) handle1 handle2
    {-# INLINE greater #-}
    greater' sym scalar = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._greater_scalar (name1 <> ">" <> show scalar) handle (realToFrac scalar)
    {-# INLINE greater' #-}
    greaterEqual sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_greater_equal (name1 <> ">=" <> name2) handle1 handle2
    {-# INLINE greaterEqual #-}
    greaterEqual' sym scalar = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._greater_equal_scalar (name1 <> ">=" <> show scalar) handle (realToFrac scalar)
    {-# INLINE greaterEqual' #-}
    lesser sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_lesser (name1 <> "<" <> name2) handle1 handle2
    {-# INLINE lesser #-}
    lesser' sym scalar = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._lesser_scalar (name1 <> "<" <> show scalar) handle (realToFrac scalar)
    {-# INLINE lesser' #-}
    lesserEqual sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_lesser_equal (name1 <> "<=" <> name2) handle1 handle2
    {-# INLINE lesserEqual #-}
    lesserEqual' sym scalar = Symbol . unsafePerformIO $ do
        let handle = getHandle sym
        name1 <- getName sym
        I._lesser_equal_scalar (name1 <> "<=" <> show scalar) handle (realToFrac scalar)
    {-# INLINE lesserEqual' #-}

instance Neural Symbol where
    fullyconnected input weight bias n = Symbol . unsafePerformIO $ do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        I.fullyconnected "FullyConnected" handle1 handle2 handle3 n nil
    convolution input weight bias kernel n = Symbol . unsafePerformIO $ do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        I.convolution "Convolution" handle1 handle2 handle3 kernel n nil
    activation input act = Symbol . unsafePerformIO $ do
        let handle1 = getHandle input
        I.activation "Activation" handle1 act
    batchnorm input weight bias = Symbol . unsafePerformIO $ do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        I.batchnorm "BatchNorm" handle1 handle2 handle3 nil
    pooling input kernel pooltype = Symbol . unsafePerformIO $ do
        let handle1 = getHandle input
        I.pooling "Pooling" handle1 kernel pooltype nil
    softmaxoutput input label = Symbol . unsafePerformIO $ do
        let handle1 = getHandle input
            handle2 = getHandle label
        I.softmaxoutput "SoftmaxOutput" handle1 handle2 nil
