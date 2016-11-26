-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Interfaces in core module of MXNet.
--
module MXNet.Core.Base (
      -- * Re-export data type definitions.
      -- ** Type alias
      MXUint
    , MXFloat
      -- ** Handlers and Creators.
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
      -- ** Callback types.
    , ExecutorMonitorCallback
    , CustomOpPropCreator
      -- * Re-export basic functions.
    , mxGetLastError
    , mxRandomSeed
    , mxNotifyShutdown
    , mxNDArrayCreateNone
    , mxNDArrayCreate
    , mxNDArrayCreateEx
    , mxNDArrayLoadFromRawBytes
    , mxNDArraySaveRawBytes
    , mxNDArraySave
    , mxNDArrayLoad
    , mxNDArraySyncCopyFromCPU
    , mxNDArraySyncCopyToCPU
    , mxNDArrayWaitToRead
    , mxNDArrayWaitToWrite
    , mxNDArrayWaitAll
    , mxNDArrayFree
    , mxNDArraySlice
    , mxNDArrayAt
    , mxNDArrayReshape
    , mxNDArrayGetShape
    , mxNDArrayGetData
    , mxNDArrayGetDType
    , mxNDArrayGetContext
    , mxListFunctions
    , mxGetFunction
    , mxFuncGetInfo
    , mxFuncDescribe
    , mxFuncInvoke
    , mxFuncInvokeEx
    , mxSymbolListAtomicSymbolCreators
    , mxSymbolGetAtomicSymbolName
    ) where

import MXNet.Core.Internal.Raw
