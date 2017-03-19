-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.DType
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- DType corresponding between Haskell's data type and numpy's data type.
--
{-# OPTIONS_GHC -Wno-missing-signatures #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE KindSignatures #-}

module MXNet.Core.Base.DType
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
    typeid _ = FLOAT64
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
    -- | Dot product.
    dot :: DType a => tensor a -> tensor a -> tensor a
    -- | Reshape a tensor value.
    reshape :: DType a => tensor a -> [Int] -> tensor a
    -- | Transpose a tensor value.
    transpose :: DType a => tensor a -> tensor a
    -- | Ordinary arithmetic operators with scalar value.
    (.+), (.-), (.*), (./), (.^) :: DType a => tensor a -> a -> tensor a
    -- | Flip version of ordinary arithmetic operators with scalar value.
    (..-), (../), (..^) :: DType a => a -> tensor a -> tensor a
    -- | Mutable ordinary arithmetic operators with scalar value.
    (.+=), (.-=), (.*=), (./=), (.^=) :: DType a => tensor a -> a -> IO ()
    -- | Compare two tensor values, after comparison, all cell may be set as a same value, or /0/, or /1/.
    _Maximum, _Minimum, equal, notEqual, greater, greaterEqual, lesser, lesserEqual
        :: DType a => tensor a -> tensor a -> tensor a
    -- | Compare a tensor value with a scalar value, after comparison, all cell may be set as a same value, or /0/, or /1/.
    _Maximum', _Minimum', equal', notEqual', greater', greaterEqual', lesser', lesserEqual'
        :: DType a => tensor a -> a -> tensor a

infixl 6 .+, .-, ..-
infixl 7 .*, ./, ../
infixr 8 .^, ..^

-- | Neural network combinators.
class Tensor tensor => Neural tensor where
    -- | Apply a linear transformation: /Y = X W^T + b/.
    fullyConnected
        :: DType a
        => tensor a     -- ^ Input data.
        -> tensor a     -- ^ Weight matrix.
        -> tensor a     -- ^ Bias parameter.
        -> Int          -- ^ Number of hidden nodes of the output.
        -> tensor a
    -- | Apply correlation to inputs
    correlation
        :: DType a
        => tensor a     -- ^ Input data1 to the correlation.
        -> tensor a     -- ^ Input data2 to the correlation.
        -> tensor a
    -- | ElementWise activation function.
    activation
        :: DType a
        => tensor a     -- ^ Input data to activation function.
        -> String       -- ^ Activation function to be applied, one of {'relu', 'sigmoid', 'softrelu', 'tanh'}.
        -> tensor a
    -- | Leaky ReLu activation
    --
    -- The following types are supported:
    -- 
    --      1. elu: /y = x > 0 ? x : slop * (exp(x)-1)/
    --      2. leaky: /y = x > 0 ? x : slope * x/
    --      3. prelu: same as leaky but the slope is learnable.
    --      4. rrelu: same as leaky but the slope is uniformly randomly chosen from [lower_bound, upper_bound) for
    --         training, while fixed to be (lower_bound+upper_bound)/2 for inference.
    leakyReLU
        :: DType a
        => tensor a     -- ^ Input data to activation function.
        -> String       -- ^ Activation function to be applied, one of {'elu', 'leaky', 'prelu', 'rrelu'}, default is 'leaky'.
        -> tensor a
    -- | Apply softmax activation to input.
    softmaxActivation
        :: DType a
        => tensor a     -- ^ Input data to activation function.
        -> tensor a
    -- | Apply dropout to input.
    dropout
        :: DType a
        => tensor a     -- ^ Input data to dropout.
        -> Float        -- ^ Fraction of the input that gets dropped out at training time, default is 0.5.
        -> tensor a
    -- | Batch normalization.
    batchNorm
        :: DType a
        => tensor a     -- ^ Input data to batch normalization.
        -> tensor a     -- ^ Gamma array.
        -> tensor a     -- ^ Beta array.
        -> tensor a
    -- | An operator taking in a n-dimensional input tensor (n > 2), and normalizing the input by subtracting the mean
    -- and variance calculated over the spatial dimensions.
    instanceNorm
        :: DType a
        => tensor a     -- ^ A n-dimensional tensor (n > 2) of the form [batch, channel, spatial_dim1, spatial_dim2, ...].
        -> tensor a     -- ^ Gamma, a vector of length 'channel', which multiplies the normalized input.
        -> tensor a     -- ^ Beta, a vector of length 'channel', which is added to the product of the normalized input and the weight.
        -> Float        -- ^ Epsilon to prevent division by 0.
        -> tensor a
    -- | Set the l2 norm of each instance to a constant.
    l2Normalization
        :: DType a
        => tensor a     -- ^ Input data to the L2NormalizationOp.
        -> Float        -- ^ Epsilon to prevent div 0, default is /1e-10/.
        -> String       -- ^ Normalization Mode, one of {'channel', 'instance', 'spatial'}, default is 'instance'.
        -> tensor a
    -- | Convolution Compute N-D convolution on (N+2)-D input.
    convolution
        :: DType a
        => tensor a     -- ^ Input data.
        -> tensor a     -- ^ Weight matrix.
        -> tensor a     -- ^ Bias parameter.
        -> String       -- ^ Convolution kernel size: (h, w) or (d, h, w).
        -> Int          -- ^ Convolution filter(channel) number.
        -> tensor a
    -- | Apply convolution to input then add a bias.
    lrn :: DType a
        => tensor a     -- ^ Input data to the ConvolutionOp.
        -> Float        -- ^ Alpha, value of the alpha variance scaling parameter in the normalization formula, default is 0.0001.
        -> Float        -- ^ Beta, value of the beta power parameter in the normalization formula, default is 0.75.
        -> Float        -- ^ Value of the k parameter in normalization formula, default is 2.
        -> Int          -- ^ Normalization window width in elements.
        -> tensor a
    -- | Apply deconvolution to input then add a bias.
    deconvolution
        :: DType a
        => tensor a     -- ^ Input data to the DeconvolutionOp.
        -> tensor a     -- ^ Weight matrix.
        -> tensor a     -- ^ Bias parameter.
        -> String       -- ^ Convolution kernel size: (h, w) or (d, h, w).
        -> Int          -- ^ Convolution filter(channel) number.
        -> tensor a
    -- | Perform pooling on the input.
    pooling
        :: DType a
        => tensor a     -- ^ Input data to the pooling operator.
        -> String       -- ^ Pooling kernel size: (y, x) or (d, y, x).
        -> String       -- ^ Pooling type to be applied, one of {'avg', 'max', 'sum'}.
        -> tensor a
    -- | Performs region-of-interest pooling on inputs.
    roiPooling
        :: DType a
        => tensor a     -- ^ Input data to the pooling operator, a 4D Feature maps.
        -> tensor a     -- ^ Bounding box coordinates.
        -> String       -- ^ Fix pooled size: (h, w).
        -> Int          -- ^ Ratio of input feature map height (or w) to raw image height (or w).
        -> tensor a
    -- | Apply a recurrent layer to input.
    rnn :: DType a
        => tensor a     -- ^ Input data to RNN.
        -> tensor a     -- ^ Vector of all RNN trainable parameters concatenated.
        -> tensor a     -- ^ Initial hidden state of the RNN.
        -> tensor a     -- ^ Initial cell state for LSTM networks (only for LSTM).
        -> Int          -- ^ Size of the state for each layer.
        -> Int          -- ^ Number of stacked layers.
        -> String       -- ^ The type of RNN to compute, one of {'gru', 'lstm', 'rnn_relu', 'rnn_tanh'}.
        -> tensor a
    -- | Map integer index to vector representations (embeddings).
    embedding
        :: DType a
        => tensor a     -- ^ Input data to the EmbeddingOp.
        -> tensor a     -- ^ Embedding weight matrix.
        -> Int          -- ^ Vocabulary size of the input indices.
        -> Int          -- ^ Dimension of the embedding vectors.
        -> tensor a
    -- | Apply bilinear sampling to input feature map, which is the key of “[NIPS2015] Spatial Transformer Networks” output[batch, channel, y_dst, x_dst] = G(data[batch, channel, y_src, x_src) x_dst, y_dst enumerate all spatial locations in output x_src = grid[batch, 0, y_dst, x_dst] y_src = grid[batch, 1, y_dst, x_dst] G() denotes the bilinear interpolation kernel The out-boundary points will be padded as zeros.
    bilinearSampler
        :: DType a
        => tensor a     -- ^ Input data to the BilinearsamplerOp.
        -> tensor a     -- ^ Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src.
        -> tensor a
    -- | generate sampling grid for bilinear sampling.
    gridGenerator
        :: DType a
        => tensor a     -- ^ Input data to the BilinearsamplerOp.
        -> tensor a     -- ^ Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src.
        -> tensor a
    -- | Perform nearest neighboor/bilinear up sampling to inputs
    upSampling
        :: DType a
        => [tensor a]   -- ^ Array of tensors to upsample.
        -> Int          -- ^ Up sampling scale.
        -> String       -- ^ Upsampling method, one of {'bilinear', 'nearest'}.
        -> tensor a
    -- | Apply spatial transformer to input feature map.
    spatialTransformer
        :: DType a
        => tensor a     -- ^ Input data to the SpatialTransformerOp.
        -> tensor a     -- ^ Localisation net, the output dim should be 6 when transform_type is affine. 
        -> tensor a
    -- | Use linear regression for final output, this is used on final output of a net.
    linearRegressionOutput
        :: DType a
        => tensor a     -- ^ Input data to function.
        -> tensor a     -- ^ Input label to function.
        -> tensor a
    -- | Use Logistic regression for final output, this is used on final output of a net.
    logisticRegressionOutput
        :: DType a
        => tensor a     -- ^ Input data to function.
        -> tensor a     -- ^ Input label to function.
        -> tensor a
    -- | Softmax with logit loss.
    softmaxOutput
        :: DType a
        => tensor a     -- ^ Input data.
        -> tensor a     -- ^ Ground truth label.
        -> tensor a
    -- | Use mean absolute error regression for final output, this is used on final output of a net.
    maeRegressionOutput
        :: DType a
        => tensor a     -- ^ Input data to function.
        -> tensor a     -- ^ Input label to function.
        -> tensor a
    -- | Support Vector Machine based transformation on input, backprop L2-SVM
    svmOutput
        :: DType a
        => tensor a     -- ^ Input data to svm.
        -> tensor a     -- ^ Label data.
        -> Int          -- ^ Margin, scale the DType(param_.margin) for activation size, default is 1.
        -> Float        -- ^ Regularization coefficient, Scale the coefficient responsible for balacing coefficient size and error
                        -- tradeoff, default is 1.
        -> Bool         -- ^ Use linear, if set true, uses L1-SVM objective function. Default uses L2-SVM objective, default is False.
        -> tensor a
    -- | Calculate cross_entropy(data, one_hot(label))
    softmaxCrossEntropy
        :: DType a
        => tensor a     -- ^ Input data.
        -> tensor a     -- ^ Input label.
        -> tensor a
    -- | Calculate Smooth L1 Loss(lhs, scalar)
    smoothL1
        :: DType a
        => tensor a     -- ^ Source input
        -> Float        -- ^ Scalar input.
        -> tensor a
    -- | Apply a sparse regularization to the output a sigmoid activation function.
    identityAttachKLSparsereg
        :: DType a
        => tensor a     -- ^ Input data.
        -> tensor a
    -- | Get output from a symbol and pass 1 gradient back.
    makeLoss
        :: DType a
        => tensor a     -- ^ Input data.
        -> Float        -- ^ Gradient scale as a supplement to unary and binary operators, default is 1.
        -> Float        -- ^ Valid thresh, default is 0. Regard element valid when x > valid_thresh, this is used only
                        -- in valid normalization mode.
        -> String       -- ^ Normalization, one of {'batch', 'null', 'valid'}, default is 'null'.
        -> tensor a
    -- | Get output from a symbol and pass 0 gradient back
    blockGrad
        :: DType a
        => tensor a     -- ^ The input.
        -> tensor a
    -- | Custom operator implemented in frontend.
    custom
        :: DType a
        => String     -- ^ Type of custom operator, must be registered first.
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
