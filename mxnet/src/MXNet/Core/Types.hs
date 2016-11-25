-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Types
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Data type definitions of MXNet.
--
module MXNet.Core.Types (
      -- * Type alias
      MXUint
    , MXFloat
      -- * Handlers and Creators.
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
    , OptimizerCreator
    , OptimizerHandle
      -- * Callback types.
    , ExecutorMonitorCallback
    , CustomOpPropCreator
    ) where

import MXNet.Core.Internal.Raw
