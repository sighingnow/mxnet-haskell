-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.Symbol
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Symbol module.
--
{-# OPTIONS_GHC -Wno-missing-methods #-}
{-# OPTIONS_GHC -Wno-redundant-constraints #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE DeriveGeneric #-}

module MXNet.Core.Base.Symbol where

import           Control.Exception (assert, throw)
import           Control.Monad
import           Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM
import           Data.IORef
import           Data.Monoid
import           Foreign.Ptr (nullPtr)
import           System.IO.Unsafe
import           Unsafe.Coerce (unsafeCoerce)
import           GHC.Generics

import           MXNet.Core.Base.DType
import           MXNet.Core.Base.Internal
import qualified MXNet.Core.Base.Internal.TH.Symbol as I
import           MXNet.Core.Base.HMap
import           MXNet.Core.Base.Executor hiding (getHandle)
import           MXNet.Core.Base.NDArray hiding (getHandle)
import qualified MXNet.Core.Base.NDArray as NDArray (getHandle)

-- | Type alias for variable.
newtype Symbol a = Symbol { getHandle :: SymbolHandle }
    deriving Generic

instance DType a => Show (Symbol a) where
    show sym = unsafePerformIO $ do
        str <- checked $ mxSymbolPrint (getHandle sym)
        return str

-- | Make a new symbolic variable with given name.
variable :: DType a
        => String           -- ^ Name.
        -> IO (Symbol a)    -- ^ Result variable.
variable name = do
    handle <- checked $ mxSymbolCreateVariable name
    return $ Symbol handle

-- | Get the name of a given variable.
getName :: DType a => Symbol a -> IO String
getName = mxSymbolGetName . getHandle >=> \(_, nm, _) -> return nm

-- | Get specified attribute of symbol.
getAttr :: DType a => Symbol a -> String -> IO (Maybe String)
getAttr sym key = do
    (s, success) <- checked $ mxSymbolGetAttr (getHandle sym) key
    return $ if success == 0    -- 0 when success, -1 when failure happens
                then Just s
                else Nothing

-- | Set specified attribute of symbol.
setAttr :: DType a => Symbol a -> String -> String -> IO ()
setAttr sym key value = void $ mxSymbolSetAttr (getHandle sym) key value

-- | Infer the shape of the given symbol, return the in, out and auxiliary shape size.
infershape :: DType a => Symbol a -> [String] -> IO ([[Int]], [[Int]], [[Int]])
infershape sym args = mxSymbolInferShape (getHandle sym) args [0] []

-- | Get the autodiff of current symbol.
-- This function can only be used if current symbol is a loss function.
grad :: DType a => Symbol a -> [String] -> IO (Symbol a)
grad sym args = do
    let nargs = fromIntegral (length args)
    handle <- checked $ mxSymbolGrad (getHandle sym) nargs args
    return $ Symbol handle

-- | Bind with explicit argument mapping (name -- value mapping).
bind :: DType a
     => Symbol a
     -> Context
     -> HashMap String (NDArray a)
     -> IO (Executor a)
bind sym Context{..} args = do
    inputs <- genNDArrayMapping <$> listInputs sym
    -- req_map = {'null': 0, 'write': 1, 'add': 3}
    let req_types = replicate (HM.size inputs) 1        -- use default value.
    exec <- checked $ mxExecutorBind (getHandle sym)
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
    genNDArrayMapping arg_names = HM.fromList (genfn <$> arg_names)
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
    inputs <- genNDArrayMapping <$> listInputs sym
    -- req_map = {'null': 0, 'write': 1, 'add': 3}
    let req_types = replicate (HM.size inputs) 1        -- use default value.
    exec <- checked $ mxExecutorBind (getHandle sym)
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
    genNDArrayMapping names =
        assert (length args == length names) $
            HM.fromList (zip names args)

-- | List all input arguments.
listInputs :: DType a => Symbol a -> IO [String]
listInputs sym = mxSymbolListArguments (getHandle sym)

-- | List all output results.
listOutputs :: DType a => Symbol a -> IO [String]
listOutputs sym = mxSymbolListOutputs (getHandle sym)

-- | List all auxiliary states.
listAuxiliaries :: DType a => Symbol a -> IO [String]
listAuxiliaries sym = mxSymbolListAuxiliaryStates (getHandle sym)

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
    signum = error "Unsupported operator: signum(Symbol)"
    fromInteger = error "Unsupported operator: fromInteger(Symbol)"

instance DType a => Fractional (Symbol a) where
    (/) sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Div (name1 <> "/" <> name2) handle1 handle2
    fromRational = error "Unsupported operator: fromRational(Symbol)"

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
    dot sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.dot ("dot(" <> name1 <> "," <> name2 <> ")") handle1 handle2 nil
    reshape sym sh = Symbol <$> do
        let handle = getHandle sym
            sh' = "(" <> (init . tail . show $ sh) <> ")"
        name1 <- getName sym
        I.reshape ("reshape(" <> name1 <> "," <> sh' <> ")") handle (add @"shape" sh' nil)
    transpose sym = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I.transpose ("transpose(" <> name1 <> ")") handle nil
    (+.) sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Plus (name1 <> "+" <> name2) handle1 handle2
    (-.) sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Minus (name1 <> "-" <> name2) handle1 handle2
    (*.) sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Mul (name1 <> "*" <> name2) handle1 handle2
    (/.) sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Div (name1 <> "*" <> name2) handle1 handle2
    (^.) sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Power (name1 <> "*" <> name2) handle1 handle2
    (.+) sym value = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._PlusScalar (name1 <> "+" <> show value) handle (realToFrac value)
    {-# INLINE (.+) #-}
    (.-) sym value = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._MinusScalar (name1 <> "-" <> show value) handle (realToFrac value)
    {-# INLINE (.-) #-}
    (.*) sym value = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._MulScalar (name1 <> "*" <> show value) handle (realToFrac value)
    {-# INLINE (.*) #-}
    (./) sym value = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._DivScalar (name1 <> "/" <> show value) handle (realToFrac value)
    {-# INLINE (./) #-}
    (.^) sym value = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._PowerScalar (name1 <> "^" <> show value) handle (realToFrac value)
    {-# INLINE (.^) #-}
    (..-) value sym = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._RMinusScalar (show value <> "-" <> name1) handle (realToFrac value)
    {-# INLINE (..-) #-}
    (../) value sym = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._RDivScalar (show value <> "/" <> name1) handle (realToFrac value)
    {-# INLINE (../) #-}
    (..^) value sym = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._RPowerScalar (show value <> "^" <> name1) handle (realToFrac value)
    {-# INLINE (..^) #-}

    _Maximum sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Maximum ("_Maximum(" <> name1 <> "," <> name2 <> ")") handle1 handle2
    {-# INLINE _Maximum #-}
    _Maximum' sym scalar = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._MaximumScalar ("_Maximum'(" <> name1 <> "," <> show scalar <> ")") handle (realToFrac scalar)
    {-# INLINE _Maximum' #-}
    _Minimum sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I._Minimum ("_Minimum(" <> name1 <> "," <> name2 <> ")") handle1 handle2
    {-# INLINE _Minimum #-}
    _Minimum' sym scalar = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._MinimumScalar ("_Minimum'(" <> name1 <> "," <> show scalar <> ")") handle (realToFrac scalar)
    {-# INLINE _Minimum' #-}
    equal sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_equal (name1 <> "==" <> name2) handle1 handle2
    {-# INLINE equal #-}
    equal' sym scalar = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._equal_scalar (name1 <> "==" <> show scalar) handle (realToFrac scalar)
    {-# INLINE equal' #-}
    notEqual sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_not_equal (name1 <> "/=" <> name2) handle1 handle2
    {-# INLINE notEqual #-}
    notEqual' sym scalar = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._not_equal_scalar (name1 <> "/=" <> show scalar) handle (realToFrac scalar)
    {-# INLINE notEqual' #-}
    greater sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_greater (name1 <> ">" <> name2) handle1 handle2
    {-# INLINE greater #-}
    greater' sym scalar = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._greater_scalar (name1 <> ">" <> show scalar) handle (realToFrac scalar)
    {-# INLINE greater' #-}
    greaterEqual sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_greater_equal (name1 <> ">=" <> name2) handle1 handle2
    {-# INLINE greaterEqual #-}
    greaterEqual' sym scalar = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._greater_equal_scalar (name1 <> ">=" <> show scalar) handle (realToFrac scalar)
    {-# INLINE greaterEqual' #-}
    lesser sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_lesser (name1 <> "<" <> name2) handle1 handle2
    {-# INLINE lesser #-}
    lesser' sym scalar = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._lesser_scalar (name1 <> "<" <> show scalar) handle (realToFrac scalar)
    {-# INLINE lesser' #-}
    lesserEqual sym1 sym2 = Symbol <$> do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- getName sym1
        name2 <- getName sym2
        I.broadcast_lesser_equal (name1 <> "<=" <> name2) handle1 handle2
    {-# INLINE lesserEqual #-}
    lesserEqual' sym scalar = Symbol <$> do
        let handle = getHandle sym
        name1 <- getName sym
        I._lesser_equal_scalar (name1 <> "<=" <> show scalar) handle (realToFrac scalar)
    {-# INLINE lesserEqual' #-}

-- | Provide a globally unique serial ID for each symbol.
symid :: IORef Int
symid = unsafePerformIO (newIORef 0)

-- | Generate a globally unique name for each symbol, thread safely.
naming :: String -> IO String
naming prefix = ((prefix <>) . show) <$> atomicModifyIORef symid (\a -> (a+1, a))

instance Neural Symbol where
    fullyConnected input weight bias n = Symbol <$> do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        name <- naming "FullyConnected"
        I.fullyconnected name handle1 handle2 handle3 n nil
    correlation input1 input2 = Symbol <$> do
        let handle1 = getHandle input1
            handle2 = getHandle input2
        name <- naming "Correlation"
        I.correlation name handle1 handle2 nil
    activation input act = Symbol <$> do
        let handle1 = getHandle input
        name <- naming "Activation"
        I.activation name handle1 act
    leakyReLU input act = Symbol <$> do
        let handle1 = getHandle input
        name <- naming "LeakyReLU"
        I.leakyrelu name handle1 (add @"act_type" act nil)
    softmaxActivation input = Symbol <$> do
        let handle1 = getHandle input
        name <- naming "SoftmaxActivation"
        I.softmaxactivation name handle1 nil
    dropout input p = Symbol <$> do
        let handle1 = getHandle input
        name <- naming "Dropout"
        I.dropout name handle1 (add @"p" p nil)
    batchNorm input gm bt mm mv = Symbol <$> do
        let handle1 = getHandle input
        let handle2 = getHandle gm
        let handle3 = getHandle bt
        let handle4 = getHandle mm
        let handle5 = getHandle mv
        name <- naming "BatchNorm"
        I.batchnorm name handle1 handle2 handle3 handle4 handle5 nil
    instanceNorm input gamma beta eps = Symbol <$> do
        let handle1 = getHandle input
            handle2 = getHandle gamma
            handle3 = getHandle beta
        name <- naming "InstnaceNorm"
        I.instancenorm name handle1 handle2 handle3 (add @"eps" eps nil)
    l2Normalization input eps mode = Symbol <$> do
        let handle1 = getHandle input
        name <- naming "L2Normalization"
        I.l2normalization name handle1 (add @"eps" eps $ add @"mode" mode nil)
    convolution input weight bias kernel n = Symbol <$> do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        name <- naming "Convolution"
        I.convolution name handle1 handle2 handle3 kernel n nil
    lrn input alpha beta knorm nsize = Symbol <$> do
        let handle1 = getHandle input
        name <- naming "LRN"
        I.lrn name handle1 nsize (add @"alpha" alpha $ add @"beta" beta $ add @"knorm" knorm nil)
    deconvolution input weight bias kernel nfilter = Symbol <$> do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        name <- naming "Deconvolution"
        I.deconvolution name handle1 handle2 handle3 kernel nfilter nil
    pooling input kernel pooltype = Symbol <$> do
        let handle1 = getHandle input
        name <- naming "Pooling"
        I.pooling name handle1 kernel pooltype nil
    softmaxOutput input label = Symbol <$> do
        let handle1 = getHandle input
            handle2 = getHandle label
        name <- naming "SoftmaxOutput"
        I.softmaxoutput name handle1 handle2 nil
    makeLoss input grad_scale valid_thresh normalization = Symbol <$> do
        let handle1 = getHandle input
        name <- naming "MakeLoss"
        I.makeloss name handle1 (add @"grad_scale" grad_scale $ add @"valid_thresh" valid_thresh $ add @"normalization" normalization nil)
    blockGrad input = Symbol <$> do
        let handle1 = getHandle input
        name <- naming "BlockGrad"
        I.blockgrad name handle1
    custom input op = Symbol <$> do
        let handles = map getHandle input
        name <- naming "Custom"
        I.custom name handles op
