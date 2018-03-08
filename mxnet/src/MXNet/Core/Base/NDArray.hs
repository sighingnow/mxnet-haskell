-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.NDArray
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- NDArray module, provide an imperative-style programming interface.
--
{-# OPTIONS_GHC -Wno-missing-methods #-}
{-# OPTIONS_GHC -Wno-redundant-constraints #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DeriveGeneric #-}

module MXNet.Core.Base.NDArray where

import           Control.Monad
import           Data.Int
import           Data.Monoid
import           Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import           Foreign.Marshal.Alloc (alloca)
import           Foreign.Marshal.Array (peekArray)
import           Foreign.Ptr
import           GHC.Exts (IsList(..))
import           Text.PrettyPrint.Annotated.HughesPJClass (Pretty(..), prettyShow)
import           System.IO.Unsafe (unsafePerformIO)
import           GHC.Generics

import           MXNet.Core.Base.DType
import           MXNet.Core.Base.Internal
import qualified MXNet.Core.Base.Internal.TH.NDArray as I
import           MXNet.Core.Base.HMap

-- | NDArray type alias.
newtype NDArray a = NDArray { getHandle :: NDArrayHandle }
    deriving Generic

-- | Wait all async operation to finish in MXNet.
waitAll :: IO ()
waitAll = void mxNDArrayWaitAll

-- | Make a new empty ndarray with specified shape, context and data type.
makeEmptyNDArray :: forall a. DType a
                 => [Int]           -- ^ Shape.
                 -> Context         -- ^ Context/
                 -> Bool            -- ^ If delayed allocate.
                 -> IO (NDArray a)
makeEmptyNDArray sh ctx delayed = do
    let sh' = fromIntegral <$> sh
        nlen = fromIntegral . length $ sh
        dtype = typeid (undefined :: a)
    handle <- checked $ mxNDArrayCreateEx sh' nlen (deviceType ctx) (deviceId ctx) (if delayed then 1 else 0) dtype
    return $ NDArray handle

-- | Make a new NDArray with given shape.
makeNDArray :: DType a
            => [Int]            -- ^ size of every dimensions.
            -> Context
            -> Vector a
            -> IO (NDArray a)
makeNDArray sh ctx ds = do
    let sh' = fromIntegral <$> sh
        nlen = fromIntegral . length $ sh
    handle <- checked $ mxNDArrayCreate sh' nlen (deviceType ctx) (deviceId ctx) 0
    V.unsafeWith ds $ \p -> do
        let len = fromIntegral (V.length ds)
        void $ mxNDArraySyncCopyFromCPU handle (castPtr p) len
        return $ NDArray handle

-- | Get the shape of given NDArray.
ndshape :: DType a
      => NDArray a
      -> IO (Int, [Int])  -- ^ Dimensions and size of every dimensions.
ndshape arr = do
    (nlen, sh) <- mxNDArrayGetShape (getHandle arr)
    return (fromIntegral nlen, fromIntegral <$> sh)

-- | Get size of the given ndarray.
ndsize :: DType a
     => NDArray a
     -> IO Int      -- ^ Dimensions and size of every dimensions.
ndsize arr = (product . snd) <$> ndshape arr

-- | Get context of the given ndarray.
context :: DType a => NDArray a -> IO Context
context arr = do
    (device'type, device'id) <- checked $ mxNDArrayGetContext (getHandle arr)
    return $ Context device'type device'id

-- | Make a copy of the give ndarray.
copy :: DType a => NDArray a -> IO (NDArray a)
copy arr = NDArray <$> I._copy (getHandle arr)

-- | Get data stored in NDArray.
items :: DType a => NDArray a -> IO (Vector a)
items arr = do
    nlen <- ndsize arr
    alloca $ \p -> do
        checked $ mxNDArraySyncCopyToCPU (getHandle arr) p (fromIntegral nlen)
        fromList <$> peekArray nlen (castPtr p :: Ptr a)

-- | Return a sliced ndarray that __shares memory__ with current one.
slice :: DType a
      => NDArray a
      -> Int        -- ^ The beginning index of slice.
      -> Int        -- ^ The end index of slices.
      -> NDArray a
slice arr start end = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    handle' <- checked $ mxNDArraySlice handle (fromIntegral start) (fromIntegral end)
    return handle'

-- | Return a sub ndarray that __shares memory__ with current one.
at :: DType a
   => NDArray a
   -> Int       -- ^ The index.
   -> NDArray a
at arr idx = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    handle' <- checked $ mxNDArrayAt handle (fromIntegral idx)
    return handle'

-- | Block until all pending writes operations on current ndarray are finished.
waitToRead :: DType a => NDArray a -> IO ()
waitToRead arr = void $ mxNDArrayWaitToRead (getHandle arr)

-- | One hot encoding indices into matrix out.
onehotEncode :: DType a
             => NDArray a       -- ^ An ndarray containing indices of the categorical features.
             -> NDArray a       -- ^ The result holder of the encoding.
             -> IO (NDArray a)  -- ^ The encoding ndarray.
onehotEncode indices out = do
    let handle1 = getHandle indices
        handle2 = getHandle out
    NDArray <$> I._onehot_encode' handle1 handle2 [handle2]

-- | Create a new NDArray filled with 0, with specified shape and context.
zeros :: DType a
      => [Int]      -- ^ Shape.
      -> IO (NDArray a)
zeros sh = full sh 0

-- | Create a new NDArray filled with 1, with specified shape and context.
ones :: DType a
     => [Int]      -- ^ Shape.
     -> IO (NDArray a)
ones sh = full sh 1

-- | Create a new NDArray filled with given value, with specified shape and context.
full :: DType a
      => [Int]      -- ^ Shape.
      -> a          -- ^ Given value to fill the ndarray.
      -> IO (NDArray a)
full sh value = makeNDArray sh contextCPU $ V.replicate (product sh) value

-- | Create a new NDArray that copies content from source_array.
array :: DType a
      => [Int]      -- ^ Shape.
      -> Vector a
      -> IO (NDArray a)
array sh = makeNDArray sh contextCPU

instance {-# OVERLAPPABLE #-} (DType a, Floating a) => Eq (NDArray a) where
    (==) arr1 arr2 = unsafePerformIO $ do
        (_, sh1) <- ndshape arr1
        (_, sh2) <- ndshape arr2
        if sh1 == sh2
            then do
                r <- (abs (arr1 - arr2) `lesser`) =<< full sh1 0.0001
                V.all (== fromIntegral (1 :: Int)) <$> items r
            else return False

instance (DType a, a ~ Int8) => Eq (NDArray Int8) where
    (==) arr1 arr2 = unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        let cmp = V.all (== fromIntegral (1 :: Int)) :: Vector a -> Bool
        (cmp <$>) . items . NDArray =<< I.broadcast_equal handle1 handle2

instance (DType a, a ~ Int32) => Eq (NDArray Int32) where
    (==) arr1 arr2 = unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        let cmp = V.all (== fromIntegral (1 :: Int)) :: Vector a -> Bool
        (cmp <$>) . items . NDArray =<< I.broadcast_equal handle1 handle2

-- | Wrapper for pretty print multiple dimensions matrices.
data PrettyWrapper = forall a. Pretty a => MkPretty { runPretty :: a }

-- | Destruct pretty
instance Pretty PrettyWrapper where
    pPrint (MkPretty inner) = pPrint inner

instance (DType a, Pretty a) => Show (NDArray a) where
    -- TODO display more related information.
    show arr = unsafePerformIO $ do
        (_, dims) <- ndshape arr
        values <- items arr
        let info = show dims
            body = prettyShow . splitItems values dims $ 0
        return ("NDArray " <> info <> "\n" <> body)
      where
        splitItems :: Vector a -> [Int] -> Int -> PrettyWrapper
        splitItems _ [] _ = error "Impossible: never match an empty list."
        splitItems values [x] s = MkPretty . toList $ V.unsafeSlice s x values
        splitItems values (d:ds) s = MkPretty $ (\x -> splitItems values ds (s + (product ds) * x)) <$> ([0 .. (d - 1)] :: [Int])

instance DType a => Num (NDArray a) where
    (+) arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_add handle1 handle2
    (-) arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_sub handle1 handle2
    (*) arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_mul handle1 handle2
    abs arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.abs handle1
    negate arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.negative handle1
    signum = error "Unsupported operator: signum(NDArray)"
    fromInteger = error "Unsupported operator: fromInteger(NDArray)"

instance DType a => Fractional (NDArray a) where
    (/) arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_div handle1 handle2
    fromRational = error "Unsupported operator: fromRational(NDArray)"

instance DType a => Floating (NDArray a) where
    exp arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.exp handle1
    log arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.log handle1
    sqrt arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.sqrt handle1
    sin arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.sin handle1
    cos arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.cos handle1
    tan arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.tan handle1
    sinh arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.sinh handle1
    cosh arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.cosh handle1
    tanh arr1 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
        I.tanh handle1

instance Tensor NDArray where
    dot arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.dot handle1 handle2 nil
    reshape arr sh = NDArray <$> do
        let handle = getHandle arr
        (_, handle') <- mxNDArrayReshape handle (length sh) sh
        return handle'
    transpose arr = NDArray <$> do
        let handle = getHandle arr
        I.transpose handle nil
    (+.) arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_add handle1 handle2
    (-.) arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_sub handle1 handle2
    (*.) arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_mul handle1 handle2
    (/.) arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_div handle1 handle2
    (^.) arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_power handle1 handle2
    (.+) arr value = NDArray <$> do
        let handle = getHandle arr
        I._plus_scalar handle (realToFrac value)
    {-# INLINE (.+) #-}
    (.-) arr value = NDArray <$> do
        let handle = getHandle arr
        I._minus_scalar handle (realToFrac value)
    {-# INLINE (.-) #-}
    (.*) arr value = NDArray <$> do
        let handle = getHandle arr
        I._mul_scalar handle (realToFrac value)
    {-# INLINE (.*) #-}
    (./) arr value = NDArray <$> do
        let handle = getHandle arr
        I._div_scalar handle (realToFrac value)
    {-# INLINE (./) #-}
    (.^) arr value = NDArray <$> do
        let handle = getHandle arr
        I._power_scalar handle (realToFrac value)
    {-# INLINE (.^) #-}

    (..-) value arr = NDArray <$> do
        let handle = getHandle arr
        I._rminus_scalar handle (realToFrac value)
    {-# INLINE (..-) #-}
    (../) value arr = NDArray <$> do
        let handle = getHandle arr
        I._rdiv_scalar handle (realToFrac value)
    {-# INLINE (../) #-}
    (..^) value arr = NDArray <$> do
        let handle = getHandle arr
        I._rpower_scalar handle (realToFrac value)
    {-# INLINE (..^) #-}

    (.+=) arr value = do
        let handle = getHandle arr
        I._plus_scalar' handle (realToFrac value) [handle]
    (.-=) arr value = do
        let handle = getHandle arr
        I._minus_scalar' handle (realToFrac value) [handle]
    (.*=) arr value = do
        let handle = getHandle arr
        I._mul_scalar' handle (realToFrac value) [handle]
    (./=) arr value = do
        let handle = getHandle arr
        I._div_scalar' handle (realToFrac value) [handle]
    (.^=) arr value = do
        let handle = getHandle arr
        I._power_scalar' handle (realToFrac value) [handle]

    _Maximum arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_maximum handle1 handle2
    {-# INLINE _Maximum #-}
    _Maximum' arr scalar = NDArray <$> do
        let handle = getHandle arr
        I._maximum_scalar handle (realToFrac scalar)
    {-# INLINE _Maximum' #-}
    _Minimum arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_minimum handle1 handle2
    {-# INLINE _Minimum #-}
    _Minimum' arr scalar = NDArray <$> do
        let handle = getHandle arr
        I._minimum_scalar handle (realToFrac scalar)
    {-# INLINE _Minimum' #-}
    equal arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_equal handle1 handle2
    {-# INLINE equal #-}
    equal' arr scalar = NDArray <$> do
        let handle = getHandle arr
        I._equal_scalar handle (realToFrac scalar)
    {-# INLINE equal' #-}
    notEqual arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_not_equal handle1 handle2
    {-# INLINE notEqual #-}
    notEqual' arr scalar = NDArray <$> do
        let handle = getHandle arr
        I._not_equal_scalar handle (realToFrac scalar)
    {-# INLINE notEqual' #-}
    greater arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_greater handle1 handle2
    {-# INLINE greater #-}
    greater' arr scalar = NDArray <$> do
        let handle = getHandle arr
        I._greater_scalar handle (realToFrac scalar)
    {-# INLINE greater' #-}
    greaterEqual arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_greater_equal handle1 handle2
    {-# INLINE greaterEqual #-}
    greaterEqual' arr scalar = NDArray <$> do
        let handle = getHandle arr
        I._greater_equal_scalar handle (realToFrac scalar)
    {-# INLINE greaterEqual' #-}
    lesser arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_lesser handle1 handle2
    {-# INLINE lesser #-}
    lesser' arr scalar = NDArray <$> do
        let handle = getHandle arr
        I._lesser_scalar handle (realToFrac scalar)
    {-# INLINE lesser' #-}
    lesserEqual arr1 arr2 = NDArray <$> do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_lesser_equal handle1 handle2
    {-# INLINE lesserEqual #-}
    lesserEqual' arr scalar = NDArray <$> do
        let handle = getHandle arr
        I._lesser_equal_scalar handle (realToFrac scalar)
    {-# INLINE lesserEqual' #-}

instance Neural NDArray where
    fullyConnected input weight bias n = NDArray <$> do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        I.fullyconnected handle1 handle2 handle3 n nil
    correlation input1 input2 = NDArray <$> do
        let handle1 = getHandle input1
            handle2 = getHandle input2
        I.correlation handle1 handle2 nil
    activation input act = NDArray <$> do
        let handle1 = getHandle input
        I.activation handle1 act
    leakyReLU input act = NDArray <$> do
        let handle1 = getHandle input
        I.leakyrelu handle1 (add @"act_type" act nil)
    softmaxActivation input = NDArray <$> do
        let handle1 = getHandle input
        I.softmaxactivation handle1 nil
    dropout input p = NDArray <$> do
        let handle1 = getHandle input
        I.dropout handle1 (add @"p" p nil)
    batchNorm input gm bt mm mv = NDArray <$> do
        let handle1 = getHandle input
        let handle2 = getHandle gm
        let handle3 = getHandle bt
        let handle4 = getHandle mm
        let handle5 = getHandle mv
        I.batchnorm handle1 handle2 handle3 handle4 handle5 nil
    instanceNorm input gamma beta eps = NDArray <$> do
        let handle1 = getHandle input
            handle2 = getHandle gamma
            handle3 = getHandle beta
        I.instancenorm handle1 handle2 handle3 (add @"eps" eps nil)
    l2Normalization input eps mode = NDArray <$> do
        let handle1 = getHandle input
        I.l2normalization handle1 (add @"eps" eps $ add @"mode" mode nil)
    convolution input weight bias kernel n = NDArray <$> do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        I.convolution handle1 handle2 handle3 kernel n nil
    lrn input alpha beta knorm nsize = NDArray <$> do
        let handle1 = getHandle input
        I.lrn handle1 nsize (add @"alpha" alpha $ add @"beta" beta $ add @"knorm" knorm nil)
    deconvolution input weight bias kernel nfilter = NDArray <$> do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        I.deconvolution handle1 handle2 handle3 kernel nfilter nil
    pooling input kernel pooltype = NDArray <$> do
        let handle1 = getHandle input
        I.pooling handle1 kernel pooltype nil
    -- roiPooling
    -- rnn
    -- embedding
    -- bilinearSampler
    -- gridGenerator
    -- upSampling
    -- spatialTransformer
    -- linearRegressionOutput
    -- logisticRegressionOutput
    softmaxOutput input label = NDArray <$> do
        let handle1 = getHandle input
            handle2 = getHandle label
        I.softmaxoutput handle1 handle2 nil
    -- maeRegressionOutput
    -- svmOutput
    -- softmaxCrossEntropy
    -- smoothL1
    -- identityAttachKLSparsereg
    makeLoss input grad_scale valid_thresh normalization = NDArray <$> do
        let handle1 = getHandle input
        I.makeloss handle1 (add @"grad_scale" grad_scale $ add @"valid_thresh" valid_thresh $ add @"normalization" normalization nil)
    blockGrad input = NDArray <$> do
        let handle1 = getHandle input
        I.blockgrad handle1
    custom input op = NDArray <$> do
        let handles = map getHandle input
        I.custom handles op
