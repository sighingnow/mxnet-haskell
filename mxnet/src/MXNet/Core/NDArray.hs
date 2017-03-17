-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.NDArray
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- NDArray module, provide an imperative-style programming interface.
--
{-# OPTIONS_GHC -Wno-orphans #-}
{-# OPTIONS_GHC -Wno-redundant-constraints #-}
{-# OPTIONS_GHC -Wno-unused-do-bind #-}

{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}

module MXNet.Core.NDArray where

import           Control.Monad
import           Data.Int
import           Data.Monoid
import           Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import           Foreign.Marshal.Alloc (alloca)
import           Foreign.Marshal.Array (peekArray)
import           Foreign.Ptr
import           Foreign.Storable (Storable)
import           GHC.Exts (IsList(..))
import           Text.PrettyPrint.Annotated.HughesPJClass (Pretty(..), prettyShow)
import           System.IO.Unsafe (unsafePerformIO)

import           MXNet.Core.Base
import           MXNet.Core.DType
import qualified MXNet.Core.Base.Internal.TH.NDArray as I
import           MXNet.Core.HMap

-- | NDArray type alias.
newtype NDArray a = NDArray { getHandle :: NDArrayHandle }

-- | Wrapper for pretty print multiple dimensions matrices.
data PrettyWrapper = forall a. Pretty a => MkPretty { runPretty :: a }

-- | Destruct pretty
instance Pretty PrettyWrapper where
    pPrint (MkPretty inner) = pPrint inner

instance (DType a, Pretty a) => Show (NDArray a) where
    -- TODO display more related information.
    show array = unsafePerformIO $ do
        (_, dims) <- shape array
        values <- items array
        let info = show dims
            body = prettyShow . splitItems values dims $ 0
        return ("NDArray " <> info <> "\n" <> body)
      where
        splitItems :: Vector a -> [Int] -> Int -> PrettyWrapper
        splitItems _ [] _ = error "Impossible: never match an empty list."
        splitItems values [x] s = MkPretty . toList $ V.unsafeSlice s x values
        splitItems values (d:ds) s = MkPretty $ (\x -> splitItems values ds (s + (product ds) * x)) <$> ([0 .. (d - 1)] :: [Int])

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
    (_, handle) <- mxNDArrayCreateEx sh' nlen (deviceType ctx) (deviceId ctx) (if delayed then 1 else 0) dtype
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
    (_, handle) <- mxNDArrayCreate sh' nlen (deviceType ctx) (deviceId ctx) 0
    V.unsafeWith ds $ \p -> do
        let len = fromIntegral (V.length ds)
        mxNDArraySyncCopyFromCPU handle (castPtr p) len
    return $ NDArray handle

-- | Get the shape of given NDArray.
shape :: DType a
      => NDArray a
      -> IO (Int, [Int])  -- ^ Dimensions and size of every dimensions.
shape arr = do
    (_, nlen, sh) <- mxNDArrayGetShape (getHandle arr)
    return (fromIntegral nlen, fromIntegral <$> sh)

-- | Get size of the given ndarray.
size :: DType a
     => NDArray a
     -> IO Int      -- ^ Dimensions and size of every dimensions.
size arr = (product . snd) <$> shape arr

-- | Get context of the given ndarray.
context :: DType a => NDArray a -> IO Context
context arr = do
    (_, device'type, device'id) <- mxNDArrayGetContext (getHandle arr)
    return $ Context device'type device'id

-- | Make a copy of the give ndarray.
copy :: DType a => NDArray a -> IO (NDArray a)
copy arr = NDArray <$> I._copy (getHandle arr)

-- | Get data stored in NDArray.
items :: DType a => NDArray a -> IO (Vector a)
items arr = do
    nlen <- (product . snd) <$> shape arr
    alloca $ \p -> do
        mxNDArraySyncCopyToCPU (getHandle arr) p (fromIntegral nlen)
        fromList <$> peekArray nlen (castPtr p :: Ptr a)

-- | Return a sliced ndarray that __shares memory__ with current one.
slice :: DType a
      => NDArray a
      -> Int        -- ^ The beginning index of slice.
      -> Int        -- ^ The end index of slices.
      -> NDArray a
slice arr start end = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    (_, handle') <- mxNDArraySlice handle (fromIntegral start) (fromIntegral end)
    return handle'

-- | Return a sub ndarray that __shares memory__ with current one.
at :: DType a
   => NDArray a
   -> Int       -- ^ The index.
   -> NDArray a
at arr idx = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    (_, handle') <- mxNDArrayAt handle (fromIntegral idx)
    return handle'

-- | Return a reshaped ndarray that __shares memory__ with current one.
reshape :: DType a
        => NDArray a
        -> [Int]        -- ^ Size of every dimension of new shape.
        -> NDArray a
reshape arr sh = NDArray . unsafePerformIO $ do
    let handle = getHandle arr
    (_, handle') <- mxNDArrayReshape handle (length sh) sh
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
full sh value = makeNDArray sh defaultContext $ V.replicate (product sh) value

-- | Create a new NDArray that copies content from source_array.
array :: DType a
      => [Int]      -- ^ Shape.
      -> Vector a
      -> IO (NDArray a)
array sh = makeNDArray sh defaultContext


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
    signum = error "Unsupported operation: signum(NDArray)"
    fromInteger = error "Unsupported operation: fromInteger(NDArray)"

instance DType a => Fractional (NDArray a) where
    (/) arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_div handle1 handle2
    fromRational = error "Unsupported operation: fromRational(NDArray)"

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
    (.+) arr value = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._plus_scalar handle (realToFrac value)
    {-# INLINE (.+) #-}
    (.-) arr value = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._minus_scalar handle (realToFrac value)
    {-# INLINE (.-) #-}
    (.*) arr value = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._mul_scalar handle (realToFrac value)
    {-# INLINE (.*) #-}
    (./) arr value = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._div_scalar handle (realToFrac value)
    {-# INLINE (./) #-}
    (.^) arr value = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._power_scalar handle (realToFrac value)
    {-# INLINE (.^) #-}

    (..-) value arr = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._rminus_scalar handle (realToFrac value)
    {-# INLINE (..-) #-}
    (../) value arr = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._rdiv_scalar handle (realToFrac value)
    {-# INLINE (../) #-}
    (..^) value arr = NDArray . unsafePerformIO $ do
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

    maximum arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_maximum handle1 handle2
    {-# INLINE maximum #-}
    maximum' arr scalar = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._maximum_scalar handle (realToFrac scalar)
    {-# INLINE maximum' #-}
    minimum arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_minimum handle1 handle2
    {-# INLINE minimum #-}
    minimum' arr scalar = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._minimum_scalar handle (realToFrac scalar)
    {-# INLINE minimum' #-}
    equal arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_equal handle1 handle2
    {-# INLINE equal #-}
    equal' arr scalar = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._equal_scalar handle (realToFrac scalar)
    {-# INLINE equal' #-}
    notEqual arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_not_equal handle1 handle2
    {-# INLINE notEqual #-}
    notEqual' arr scalar = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._not_equal_scalar handle (realToFrac scalar)
    {-# INLINE notEqual' #-}
    greater arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_greater handle1 handle2
    {-# INLINE greater #-}
    greater' arr scalar = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._greater_scalar handle (realToFrac scalar)
    {-# INLINE greater' #-}
    greaterEqual arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_greater_equal handle1 handle2
    {-# INLINE greaterEqual #-}
    greaterEqual' arr scalar = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._greater_equal_scalar handle (realToFrac scalar)
    {-# INLINE greaterEqual' #-}
    lesser arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_lesser handle1 handle2
    {-# INLINE lesser #-}
    lesser' arr scalar = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._lesser_scalar handle (realToFrac scalar)
    {-# INLINE lesser' #-}
    lesserEqual arr1 arr2 = NDArray . unsafePerformIO $ do
        let handle1 = getHandle arr1
            handle2 = getHandle arr2
        I.broadcast_lesser_equal handle1 handle2
    {-# INLINE lesserEqual #-}
    lesserEqual' arr scalar = NDArray . unsafePerformIO $ do
        let handle = getHandle arr
        I._lesser_equal_scalar handle (realToFrac scalar)
    {-# INLINE lesserEqual' #-}

instance Neural NDArray where
    fullyconnected input weight bias n = NDArray . unsafePerformIO $ do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        I.fullyconnected handle1 handle2 handle3 n nil
    convolution input weight bias kernel n = NDArray . unsafePerformIO $ do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        I.convolution handle1 handle2 handle3 kernel n nil
    activation input act = NDArray . unsafePerformIO $ do
        let handle1 = getHandle input
        I.activation handle1 act
    batchnorm input weight bias = NDArray . unsafePerformIO $ do
        let handle1 = getHandle input
            handle2 = getHandle weight
            handle3 = getHandle bias
        I.batchnorm handle1 handle2 handle3 nil
    pooling input kernel pooltype = NDArray . unsafePerformIO $ do
        let handle1 = getHandle input
        I.pooling handle1 kernel pooltype nil
    softmaxoutput input label = NDArray . unsafePerformIO $ do
        let handle1 = getHandle input
            handle2 = getHandle label
        I.softmaxoutput handle1 handle2 nil
