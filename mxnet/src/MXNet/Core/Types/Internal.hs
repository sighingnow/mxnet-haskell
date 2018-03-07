-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Types.Internal
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Types in MXNet and NNVM.
--
module MXNet.Core.Types.Internal
    ( -- * Type alias
      NNUInt
    , MXUInt
    , MXFloat
      -- * Handlers and Creators
    , NDArrayHandle
    , FunctionHandle
    , AtomicSymbolCreator
    , SymbolHandle
    , AtomicSymbolHandle
    , ExecutorHandle
    , DataIterCreator
    , DataIterHandle
    , KVStoreHandle
    , RecordIOHandle
    , RtcHandle
      -- * Handlers in predict API
    , PredictorHandle
    , NDListHandle
      -- * Handlers in nnvm
    , OpHandle
    , GraphHandle
      -- * Callback types
    , ExecutorMonitorCallback
    , CustomOpPropCreator
    , MXKVStoreUpdater
    , MXKVStoreServerController
    , nullNDArrayHandle
    ) where

import MXNet.Core.Types.Internal.Raw
import Foreign.Ptr (nullPtr)
import Foreign.ForeignPtr (newForeignPtr_)

nullNDArrayHandle :: IO NDArrayHandle
nullNDArrayHandle = NDArrayHandle <$> newForeignPtr_ nullPtr
