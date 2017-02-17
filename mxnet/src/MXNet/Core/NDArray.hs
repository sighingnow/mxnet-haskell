-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.NDArray
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- NDArray module.
--
{-# OPTIONS_GHC -Wno-orphans #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeSynonymInstances #-}

module MXNet.Core.NDArray (
      -- * Data type definitions
      NDArray (..)
    , DType (..)
    , pattern FLOAT32
    , pattern FLOAT64
    , pattern FLOAT16
    , pattern UINT8
    , pattern INT32
    , Context
      -- * Functions about NDArray
    , makeNDArray
    , getNDArrayShape
    , getNDArrayData
      -- * Default contexts
    , defaultContext
    , contextCPU
    , contextGPU
    ) where

import           Data.Array.Storable
import           Data.Int
import           Foreign.Marshal.Alloc (alloca)
import           Foreign.Marshal.Array (peekArray)
import           Foreign.Ptr
import           Foreign.Storable (Storable)
import           GHC.Exts (IsList(..))
import           System.IO.Unsafe (unsafePerformIO)

import           MXNet.Core.Base

-- | NDArray type alias.
newtype NDArray a = NDArray { getHandle :: NDArrayHandle }

-- | DType class, used to quantify types that can be passed to mxnet.
class (Storable a, Show a, Eq a) => DType a where
    typeid :: NDArray a -> Int
    typename :: NDArray a -> String

pattern FLOAT32 = 0
pattern FLOAT64 = 1
pattern FLOAT16 = 2
pattern UINT8   = 3
pattern INT32   = 4

instance DType Float where
    typeid _ = FLOAT32
    typename _ = "float32"
    {-# INLINE typeid #-}
    {-# INLINE typename #-}

instance DType Double where
    typeid _ = FLOAT32
    typename _ = "float64"
    {-# INLINE typeid #-}
    {-# INLINE typename #-}

instance DType Int8 where
    typeid _ = UINT8
    typename _ = "uint8"
    {-# INLINE typeid #-}
    {-# INLINE typename #-}

instance DType Int32 where
    typeid _ = INT32
    typename _ = "int32"
    {-# INLINE typeid #-}
    {-# INLINE typename #-}

instance DType a => Show (NDArray a) where
    -- TODO display more related information.
    show array = unsafePerformIO $ show <$> getNDArrayData array

-- | Vector type based on unboxed array.
type Vec a = StorableArray Int a

instance DType a => IsList (Vec a) where
    type (Item (Vec a)) = a
    fromList xs = unsafePerformIO (newListArray (0, length xs - 1) xs :: IO (Vec a))
    toList arr = unsafePerformIO (getElems arr :: IO [a])

instance DType a => Show (Vec a) where
    show = show . toList

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

-- | Default context, use the CPU 0 as device.
defaultContext :: Context
defaultContext = Context { deviceType = 1   -- cpu
                         , deviceId = 1     -- default value.
                         }

-- | Context for CPU 0.
contextCPU :: Context
contextCPU = Context 1 0

-- | Context for GPU 0.
contextGPU :: Context
contextGPU = Context 2 0

-- | Make a new NDArray with given shape.
makeNDArray :: DType a
            => [Int]            -- ^ size of every dimensions.
            -> Vec a
            -> IO (NDArray a)
makeNDArray shape ds = do
    let shape' = fromIntegral <$> shape
        nlen = fromIntegral . length $ shape
    (_, handle) <- mxNDArrayCreate shape' nlen (deviceType contextCPU) (deviceId contextCPU) 0
    withStorableArray ds $ \p -> do
        (l, r) <- getBounds ds
        mxNDArraySyncCopyFromCPU handle (castPtr p) (fromIntegral (r-l+1))
    return (NDArray handle)

-- | Get the shape of given NDArray.
getNDArrayShape :: NDArray a
                -> IO (Int, [Int])  -- ^ Dimensions and size of every dimensions.
getNDArrayShape array = do
    (_, nlen, shape) <- mxNDArrayGetShape (getHandle array)
    return (fromIntegral nlen, fromIntegral <$> shape)

-- | Get data stored in NDArray.
getNDArrayData :: DType a => NDArray a -> IO (Vec a)
getNDArrayData array = do
    nlen <- (product . snd) <$> getNDArrayShape array
    alloca $ \p -> do
        mxNDArraySyncCopyToCPU (getHandle array) p (fromIntegral nlen)
        fromList <$> peekArray nlen (castPtr p :: Ptr a)
