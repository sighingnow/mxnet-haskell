-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Symbol
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Symbol module.
--
{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# OPTIONS_GHC -Wno-redundant-constraints #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE RecordWildCards #-}

module MXNet.Core.Symbol where

import           Control.Exception
import           Control.Monad
import           Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM
import           Data.Monoid
import           Foreign.Ptr (nullPtr)
import           System.IO.Unsafe
import           Unsafe.Coerce (unsafeCoerce)

import           MXNet.Core.Base
import           MXNet.Core.Executor hiding (getHandle)
import           MXNet.Core.NDArray hiding (getHandle)
import qualified MXNet.Core.NDArray as NDArray (getHandle)
import           MXNet.Core.NNVM.Base
import qualified MXNet.Core.Base.Internal.TH.Symbol as I
import           MXNet.Core.HMap

-- | Type alias for variable.
newtype Symbol a = Symbol { getHandle :: SymbolHandle }

instance DType a => Show (Symbol a) where
    show sym = unsafePerformIO $ do
        (_, str) <- mxSymbolPrint (getHandle sym)
        return str

instance DType a => Num (Symbol a) where
    (+) sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- name sym1
        name2 <- name sym2
        I._Plus (name1 <> "+" <> name2) handle1 handle2
    (-) sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- name sym1
        name2 <- name sym2
        I._Minus (name1 <> "-" <> name2) handle1 handle2
    (*) sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- name sym1
        name2 <- name sym2
        I._Mul (name1 <> "*" <> name2) handle1 handle2
    abs sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.abs ("|" <> name1 <> "|") handle1
    negate sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.negative ("(-" <> name1 <> ")") handle1
    signum = error "Unsupported operation: signum(Symbol)"
    fromInteger = error "Unsupported operation: fromInteger(Symbol)"

instance DType a => Fractional (Symbol a) where
    (/) sym1 sym2 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
            handle2 = getHandle sym2
        name1 <- name sym1
        name2 <- name sym2
        I._Div (name1 <> "/" <> name2) handle1 handle2
    fromRational = error "Unsupported operation: fromRational(Symbol)"

instance DType a => Floating (Symbol a) where
    exp sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.exp ("exp(" <> name1 <> ")") handle1
    log sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.log ("log(" <> name1 <> ")") handle1
    sqrt sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.sqrt ("sqrt(" <> name1 <> ")") handle1
    sin sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.sin ("sin(" <> name1 <> ")") handle1
    cos sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.cos ("cos(" <> name1 <> ")") handle1
    tan sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.tan ("tan(" <> name1 <> ")") handle1
    sinh sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.sinh ("sinh(" <> name1 <> ")") handle1
    cosh sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.cosh ("cosh(" <> name1 <> ")") handle1
    tanh sym1 = Symbol . unsafePerformIO $ do
        let handle1 = getHandle sym1
        name1 <- name sym1
        I.tanh ("tanh(" <> name1 <> ")") handle1

-- | Make a new symbolic variable with given name.
makeVar :: DType a
        => String           -- ^ Name.
        -> IO (Symbol a)    -- ^ Result variable.
makeVar = mxSymbolCreateVariable >=> return . Symbol . snd

-- | Get the name of a given variable.
name :: DType a => Symbol a -> IO String
name = mxSymbolGetName . getHandle >=> \(_, nm, _) -> return nm

-- | Get some attribute of symbol.
attr :: DType a => Symbol a -> String -> IO (Maybe String)
attr sym key = do
    (_, s, success) <- mxSymbolGetAttr (getHandle sym) key
    return $ if success == 0    -- 0 when success, -1 when failure happens
                then Just s
                else Nothing

-- | Infer the type of the given symbol, return arg_types, out_types and aux_types.
inferType :: DType a => Symbol a -> [String] -> IO ([Int], [Int], [Int])
inferType sym args = do
    (_, arg, out, aux) <- mxSymbolInferType (getHandle sym) args
    return (arg, out, aux)

-- | Infer the shape of the given symbol, return the in, out and auxiliary shape size.
inferShape :: DType a => Symbol a -> [String] -> IO ([[Int]], [[Int]], [[Int]])
inferShape sym args = do
    (_, arg, out, aux) <- mxSymbolInferShape (getHandle sym) args [0] []
    return (arg, out, aux)

-- | List all input arguments.
getInputs :: DType a => Symbol a -> IO [String]
getInputs sym = snd <$> mxSymbolListArguments (getHandle sym)

-- | List all output results.
getOutputs :: DType a => Symbol a -> IO [String]
getOutputs sym = snd <$> mxSymbolListOutputs (getHandle sym)

-- | List all auxiliary states.
getAuxiliary :: DType a => Symbol a -> IO [String]
getAuxiliary sym = snd <$> mxSymbolListAuxiliaryStates (getHandle sym)

-- | Compose multiple symbols.
--
-- This function will change the sym hanlde. To achieve function apply behavior,
-- copy the symbol first before apply.
compose :: DType a => Symbol a -> String -> [Symbol a] -> HashMap String (Symbol a) -> IO ()
compose var name args kwargs = do
    when ((not . null) args && (not . HM.null) kwargs) $
        throwIO $ userError "compose only accept input Symbols either as positional or keyword arguments, not both"
    let (arg_names, arg_values) = if null args
                                     then (HM.keys kwargs, HM.elems kwargs)
                                     else ([], args)
    void $ nnSymbolCompose (getHandle var) name arg_names (getHandle <$> arg_values)

createAtomic :: DType a => String -> OpHandle -> [Symbol a] -> [String] -> [String] -> IO (Symbol a)
createAtomic name op args keys values = do
    let nargs = fromIntegral (length keys)
    (_, h) <- mxSymbolCreateAtomicSymbol op nargs keys values
    let sym = Symbol h
    compose sym name args [] >> return sym

softmax :: IO OpHandle
softmax = do
    (r, handle) <- nnGetOpHandle "SoftmaxOutput"
    return handle

softmaxvar :: DType a => Symbol a -> Symbol a -> IO (Symbol a)
softmaxvar a b = do
    op <- softmax
    createAtomic "softmax" op [a, b] [] []

-- | Get the autodiff of current symbol.
-- This function can only be used if current symbol is a loss function.
grad :: DType a => Symbol a -> [String] -> IO (Symbol a)
grad sym args = do
    let nargs = fromIntegral (length args)
    (_, handle) <- mxSymbolGrad (getHandle sym) nargs args
    return $ Symbol handle

-- | Bind with explicit argument mapping (name -- value mapping).
bind :: DType a
     => Symbol a
     -> Context
     -> HashMap String (NDArray a)
     -> IO (Executor a)
bind sym Context{..} args = do
    inputs <- genNDArrayMapping args <$> getInputs sym
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
    inputs <- genNDArrayMapping args <$> getInputs sym
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

-- | Add a scalar to a symbol.
(.+) :: DType a
      => Symbol a -> a -> Symbol a
(.+) sym value = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._PlusScalar (name1 <> "+" <> show value) handle (realToFrac value)

infixl 6 .+

{-# INLINE (.+) #-}

-- | Subtract a scalar from a symbol.
(.-) :: DType a
      => Symbol a -> a -> Symbol a
(.-) sym value = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._MinusScalar (name1 <> "-" <> show value) handle (realToFrac value)

infixl 6 .-

{-# INLINE (.-) #-}

-- | A symbol is subtracted by a scalar.
(..-) :: DType a
      => a -> Symbol a -> Symbol a
(..-) value sym = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._RMinusScalar (show value <> "-" <> name1) handle (realToFrac value)

infixl 6 ..-

{-# INLINE (..-) #-}

-- | Multiply a scalar to a symbol.
(.*) :: DType a
      => Symbol a -> a -> Symbol a
(.*) sym value = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._MulScalar (name1 <> "*" <> show value) handle (realToFrac value)

infixl 7 .*

{-# INLINE (.*) #-}

-- | Divide a scalar from a symbol.
(./) :: DType a
      => Symbol a -> a -> Symbol a
(./) sym value = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._DivScalar (name1 <> "/" <> show value) handle (realToFrac value)

infixl 7 ./

{-# INLINE (./) #-}

-- | Divide a scalar with a symbol.
(../) :: DType a
      => a -> Symbol a  -> Symbol a
(../) value sym = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._RDivScalar (show value <> "/" <> name1) handle (realToFrac value)

infixl 7 ../

{-# INLINE (../) #-}

-- | Power of a symbol, use a scalar as exponent.
(.^) :: DType a
      => Symbol a -> a -> Symbol a
(.^) sym value = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._PowerScalar (name1 <> "^" <> show value) handle (realToFrac value)

infixl 8 .^

{-# INLINE (.^) #-}

-- | Power of a symbol, use a symbol as exponent.
(..^) :: DType a
      => a -> Symbol a -> Symbol a
(..^) value sym = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._RPowerScalar (show value <> "^" <> name1) handle (realToFrac value)

infixl 8 ..^

{-# INLINE (..^) #-}

-- | Maximum elements in the given two symbols.
maximum :: DType a
        => Symbol a -> Symbol a -> Symbol a
maximum sym1 sym2 = Symbol . unsafePerformIO $ do
    let handle1 = getHandle sym1
        handle2 = getHandle sym2
    name1 <- name sym1
    name2 <- name sym2
    I._Maximum ("maximum(" <> name1 <> "," <> name2 <> ")") handle1 handle2

{-# INLINE maximum #-}

-- | Maximum elements in the given symbol and given scalar.
maximum' :: DType a
        => Symbol a -> a -> Symbol a
maximum' sym scalar = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._MaximumScalar ("maximum'(" <> name1 <> "," <> show scalar <> ")") handle (realToFrac scalar)

{-# INLINE maximum' #-}

-- | Minimum elements in the given two symbols.
minimum :: DType a
        => Symbol a -> Symbol a -> Symbol a
minimum sym1 sym2 = Symbol . unsafePerformIO $ do
    let handle1 = getHandle sym1
        handle2 = getHandle sym2
    name1 <- name sym1
    name2 <- name sym2
    I._Minimum ("minimum(" <> name1 <> "," <> name2 <> ")") handle1 handle2

{-# INLINE minimum #-}

-- | Minimum elements in the given symbol and given scalar.
minimum' :: DType a
        => Symbol a -> a -> Symbol a
minimum' sym scalar = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._MinimumScalar ("minimum'(" <> name1 <> "," <> show scalar <> ")") handle (realToFrac scalar)

{-# INLINE minimum' #-}

-- | If elements in the given two symbols are equal.
equal :: DType a
      => Symbol a -> Symbol a -> Symbol a
equal sym1 sym2 = Symbol . unsafePerformIO $ do
    let handle1 = getHandle sym1
        handle2 = getHandle sym2
    name1 <- name sym1
    name2 <- name sym2
    I.broadcast_equal (name1 <> "==" <> name2) handle1 handle2

{-# INLINE equal #-}

-- | If elements in the given symbol are equal to the given scalar.
equal' :: DType a
       => Symbol a -> a -> Symbol a
equal' sym scalar = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._equal_scalar (name1 <> "==" <> show scalar) handle (realToFrac scalar)

{-# INLINE equal' #-}

-- | If elements in the given two symbols are not equal.
notEqual :: DType a
          => Symbol a -> Symbol a -> Symbol a
notEqual sym1 sym2 = Symbol . unsafePerformIO $ do
    let handle1 = getHandle sym1
        handle2 = getHandle sym2
    name1 <- name sym1
    name2 <- name sym2
    I.broadcast_not_equal (name1 <> "/=" <> name2) handle1 handle2

{-# INLINE notEqual #-}

-- | If elements in the given symbol are equal to the given scalar.
notEqual' :: DType a
        => Symbol a -> a -> Symbol a
notEqual' sym scalar = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._not_equal_scalar (name1 <> "/=" <> show scalar) handle (realToFrac scalar)

{-# INLINE notEqual' #-}

-- | If elements in the first given symbols are greater than the second one.
greater :: DType a
        => Symbol a -> Symbol a -> Symbol a
greater sym1 sym2 = Symbol . unsafePerformIO $ do
    let handle1 = getHandle sym1
        handle2 = getHandle sym2
    name1 <- name sym1
    name2 <- name sym2
    I.broadcast_greater (name1 <> ">" <> name2) handle1 handle2

{-# INLINE greater #-}

-- | If elements in the first given symbols are greater than the give scalar.
greater' :: DType a
        => Symbol a -> a -> Symbol a
greater' sym scalar = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._greater_scalar (name1 <> ">" <> show scalar) handle (realToFrac scalar)

{-# INLINE greater' #-}

-- | If elements in the first given symbols are greater than or equal to the second one.
greaterEqual :: DType a
        => Symbol a -> Symbol a -> Symbol a
greaterEqual sym1 sym2 = Symbol . unsafePerformIO $ do
    let handle1 = getHandle sym1
        handle2 = getHandle sym2
    name1 <- name sym1
    name2 <- name sym2
    I.broadcast_greater_equal (name1 <> ">=" <> name2) handle1 handle2

{-# INLINE greaterEqual #-}

-- | If elements in the first given symbols are greater than or equal to the give scalar.
greaterEqual' :: DType a
              => Symbol a -> a -> Symbol a
greaterEqual' sym scalar = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._greater_equal_scalar (name1 <> ">=" <> show scalar) handle (realToFrac scalar)

{-# INLINE greaterEqual' #-}

-- | If elements in the first given symbols are lesser than the second one.
lesser :: DType a
       => Symbol a -> Symbol a -> Symbol a
lesser sym1 sym2 = Symbol . unsafePerformIO $ do
    let handle1 = getHandle sym1
        handle2 = getHandle sym2
    name1 <- name sym1
    name2 <- name sym2
    I.broadcast_lesser (name1 <> "<" <> name2) handle1 handle2

{-# INLINE lesser #-}

-- | If elements in the first given symbols are lesser than the give scalar.
lesser' :: DType a
        => Symbol a -> a -> Symbol a
lesser' sym scalar = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._lesser_scalar (name1 <> "<" <> show scalar) handle (realToFrac scalar)

{-# INLINE lesser' #-}

-- | If elements in the first given symbols are lesser than or equal to the second one.
lesserEqual :: DType a
            => Symbol a -> Symbol a -> Symbol a
lesserEqual sym1 sym2 = Symbol . unsafePerformIO $ do
    let handle1 = getHandle sym1
        handle2 = getHandle sym2
    name1 <- name sym1
    name2 <- name sym2
    I.broadcast_lesser_equal (name1 <> "<=" <> name2) handle1 handle2

{-# INLINE lesserEqual #-}

-- | If elements in the first given symbols are lesser than or equal to the give scalar.
lesserEqual' :: DType a
             => Symbol a -> a -> Symbol a
lesserEqual' sym scalar = Symbol . unsafePerformIO $ do
    let handle = getHandle sym
    name1 <- name sym
    I._lesser_equal_scalar (name1 <> "<=" <> show scalar) handle (realToFrac scalar)

{-# INLINE lesserEqual' #-}

{-------------------------------------------------------------------------------
-- Neural Network
-------------------------------------------------------------------------------}

-- | Apply a linear transformation: /Y = X W^T + b/.
fullyconnected :: DType a
               => Symbol a -- ^ Input data.
               -> Symbol a -- ^ Weight matrix.
               -> Symbol a -- ^ Bias parameter.
               -> Int       -- ^ Number of hidden nodes of the output.
               -> Symbol a
fullyconnected input weight bias n = Symbol . unsafePerformIO $ do
    let handle1 = getHandle input
        handle2 = getHandle weight
        handle3 = getHandle bias
    I.fullyconnected "FullyConnected" handle1 handle2 handle3 n nil

-- Convolution Compute N-D convolution on (N+2)-D input.
convolution :: DType a
            => Symbol a    -- ^ Input data.
            -> Symbol a    -- ^ Weight matrix.
            -> Symbol a    -- ^ Bias parameter.
            -> String       -- ^ Convolution kernel size: (h, w) or (d, h, w).
            -> Int          -- ^ Convolution filter(channel) number.
            -> Symbol a
convolution input weight bias kernel n = Symbol . unsafePerformIO $ do
    let handle1 = getHandle input
        handle2 = getHandle weight
        handle3 = getHandle bias
    I.convolution "Convolution" handle1 handle2 handle3 kernel n nil

-- | Elementwise activation function.
activation :: DType a
           => Symbol a -- ^ Input data to activation function.
           -> String    -- ^ Activation function to be applied, one of {'relu', 'sigmoid', 'softrelu', 'tanh'}.
           -> Symbol a
activation input act = Symbol . unsafePerformIO $ do
    let handle1 = getHandle input
    I.activation "Activation" handle1 act

-- | Batch normalization.
batchnorm :: DType a
          => Symbol a  -- ^ Input data to batch normalization.
          -> Symbol a  -- ^ Gamma array.
          -> Symbol a  -- ^ Beta array.
          -> Symbol a  
batchnorm input weight bias = Symbol . unsafePerformIO $ do
    let handle1 = getHandle input
        handle2 = getHandle weight
        handle3 = getHandle bias
    I.batchnorm "BatchNorm" handle1 handle2 handle3 nil

-- | Perform pooling on the input.
pooling :: DType a
        => Symbol a    -- ^ Input data to the pooling operator.
        -> String       -- ^ Pooling kernel size: (y, x) or (d, y, x).
        -> String       -- ^ Pooling type to be applied, one of {'avg', 'max', 'sum'}.
        -> Symbol a
pooling input kernel pooltype = Symbol . unsafePerformIO $ do
    let handle1 = getHandle input
    I.pooling "Pooling" handle1 kernel pooltype nil

-- | Softmax with logit loss.
softmaxoutput :: DType a
              => Symbol a  -- ^ Input data.
              -> Symbol a  -- ^ Ground truth label.
              -> Symbol a
softmaxoutput input label = Symbol . unsafePerformIO $ do
    let handle1 = getHandle input
        handle2 = getHandle label
    I.softmaxoutput "SoftmaxOutput" handle1 handle2 nil
