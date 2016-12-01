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
      -- * Data type definitions
      -- ** Type alias
      MXUInt
    , MXFloat
      -- ** Handlers and Creators
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
      -- ** Callback types
    , ExecutorMonitorCallback
    , CustomOpPropCreator
      -- * Error handling.
    , mxGetLastError
      -- * Global State setups
    , mxRandomSeed
    , mxNotifyShutdown
      -- * NDArray creation and deletion
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
      -- * Functions on NDArray
    , mxListFunctions
    , mxGetFunction
    , mxFuncGetInfo
    , mxFuncDescribe
    , mxFuncInvoke
    , mxFuncInvokeEx
      -- * Symbolic configuration generation.
    , mxSymbolListAtomicSymbolCreators
    , mxSymbolGetAtomicSymbolName
    , mxSymbolGetAtomicSymbolInfo
    , mxSymbolCreateAtomicSymbol
    , mxSymbolCreateVariable
    , mxSymbolCreateGroup
    , mxSymbolCreateFromFile
    , mxSymbolCreateFromJSON
    , mxSymbolSaveToFile
    , mxSymbolSaveToJSON
    , mxSymbolFree
    , mxSymbolCopy
    , mxSymbolPrint
    , mxSymbolGetName
    , mxSymbolGetAttr
    , mxSymbolSetAttr
    , mxSymbolListAttr
    , mxSymbolListAttrShallow
    , mxSymbolListArguments
    , mxSymbolListOutputs
    , mxSymbolGetInternals
    , mxSymbolGetOutput
    , mxSymbolListAuxiliaryStates
    ) where

import MXNet.Core.Internal.Types.Raw
import MXNet.Core.Base.Internal.Raw
