-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.DType
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- DType corresponding between Haskell's data type and numpy's data type.
--
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE KindSignatures #-}

module MXNet.Core.DType
    ( DType (..)
    , pattern FLOAT32
    , pattern FLOAT64
    , pattern FLOAT16
    , pattern UINT8
    , pattern INT32
    , Tensor (..)
    , Neural (..)
    , Context (..)
    , contextCPU
    , contextGPU
    ) where

import           Data.Int
import           Foreign.Storable (Storable)

-- | DType class, used to quantify types that can be passed to mxnet.
class (Storable a, Show a, Eq a, Ord a, Num a, Real a) => DType a where
    typeid :: a -> Int
    typename :: a -> String

pattern FLOAT32 = 0
pattern FLOAT64 = 1
pattern FLOAT16 = 2
pattern UINT8   = 3
pattern INT32   = 4

instance DType Float where
    typeid _ = FLOAT32
    {-# INLINE typeid #-}
    typename _ = "float32"
    {-# INLINE typename #-}

instance DType Double where
    typeid _ = FLOAT32
    {-# INLINE typeid #-}
    typename _ = "float64"
    {-# INLINE typename #-}

instance DType Int8 where
    typeid _ = UINT8
    {-# INLINE typeid #-}
    typename _ = "uint8"
    {-# INLINE typename #-}

instance DType Int32 where
    typeid _ = INT32
    {-# INLINE typeid #-}
    typename _ = "int32"
    {-# INLINE typename #-}

-- | Tensor operations.
class Tensor (tensor :: * -> *) where
    -- | Ordinary arithmetic operators with scalar value.
    (.+), (.-), (.*), (./), (.^) :: DType a => tensor a -> a -> tensor a
    -- | Flip version of ordinary arithmetic operators with scalar value.
    (..-), (../), (..^) :: DType a => a -> tensor a -> tensor a
    -- | Mutable ordinary arithmetic operators with scalar value.
    (.+=), (.-=), (.*=), (./=), (.^=) :: DType a => tensor a -> a -> IO ()
    -- | Compare two tensor values, after comparison, all cell may be set as a same value, or /0/, or /1/.
    maximum, minimum, equal, notEqual, greater, greaterEqual, lesser, lesserEqual
        :: DType a => tensor a -> tensor a -> tensor a
    -- | Compare a tensor value with a scalar value, after comparison, all cell may be set as a same value, or /0/, or /1/.
    maximum', minimum', equal', notEqual', greater', greaterEqual', lesser', lesserEqual'
        :: DType a => tensor a -> a -> tensor a

infixl 6 .+, .-, ..-
infixl 7 .*, ./, ../
infixr 8 .^, ..^

-- | Neural network combinators.
class Tensor tensor => Neural tensor where
    -- | Apply a linear transformation: /Y = X W^T + b/.
    fullyconnected
        :: DType a
        => tensor a    -- ^ Input data.
        -> tensor a    -- ^ Weight matrix.
        -> tensor a    -- ^ Bias parameter.
        -> Int          -- ^ Number of hidden nodes of the output.
        -> tensor a
    -- | Convolution Compute N-D convolution on (N+2)-D input.
    convolution
        :: DType a
        => tensor a    -- ^ Input data.
        -> tensor a    -- ^ Weight matrix.
        -> tensor a    -- ^ Bias parameter.
        -> String       -- ^ Convolution kernel size: (h, w) or (d, h, w).
        -> Int          -- ^ Convolution filter(channel) number.
        -> tensor a
    -- | ElementWise activation function.
    activation
        :: DType a
        => tensor a    -- ^ Input data to activation function.
        -> String       -- ^ Activation function to be applied, one of {'relu', 'sigmoid', 'softrelu', 'tanh'}.
        -> tensor a
    -- | Batch normalization.
    batchnorm
        :: DType a
        => tensor a    -- ^ Input data to batch normalization.
        -> tensor a    -- ^ Gamma array.
        -> tensor a    -- ^ Beta array.
        -> tensor a
    -- | Perform pooling on the input.
    pooling
        :: DType a
        => tensor a    -- ^ Input data to the pooling operator.
        -> String       -- ^ Pooling kernel size: (y, x) or (d, y, x).
        -> String       -- ^ Pooling type to be applied, one of {'avg', 'max', 'sum'}.
        -> tensor a
    -- | Softmax with logit loss.
    softmaxoutput
        :: DType a
        => tensor a    -- ^ Input data.
        -> tensor a    -- ^ Ground truth label.
        -> tensor a

-- | Context definition.
--
--      * DeviceType
--
--          1. cpu
--          2. gpu
--          3. cpu_pinned
data Context = Context { deviceType :: Int
                       , deviceId   :: Int
                       } deriving (Eq, Show)

-- | Context for CPU 0.
contextCPU :: Context
contextCPU = Context 1 0

-- | Context for GPU 0.
contextGPU :: Context
contextGPU = Context 2 0
