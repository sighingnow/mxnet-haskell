-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.NDArray
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- NDArray module.
--
module MXNet.Core.NDArray (
      -- * Data type definitions
      NDArray
    , Context
      -- * Functions about NDArray
    , makeNDArray
    , getNDArrayShape
      -- * Default contexts
    , defaultContext
    , contextCPU
    , contextGPU
    ) where

import           MXNet.Core.Base

-- | NDArray type alias.
type NDArray = NDArrayHandle

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
makeNDArray :: [Int]        -- ^ size of every dimensions.
            -> IO NDArray
makeNDArray shape = do
    let shape' = fromIntegral <$> shape
        nlen = fromIntegral . length $ shape
    (_, handle) <- mxNDArrayCreate shape' nlen (deviceType contextCPU) (deviceId contextCPU) 0
    return handle

-- | Get the shape of given NDArray.
getNDArrayShape :: NDArray
                -> IO (Int, [Int])  -- ^ Dimensions and size of every dimensions.
getNDArrayShape array = do
    (_, nlen, shape) <- mxNDArrayGetShape array
    return (fromIntegral nlen, fromIntegral <$> shape)
