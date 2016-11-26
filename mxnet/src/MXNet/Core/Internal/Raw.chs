-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Internal.Raw
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Direct C FFI bindings for <mxnet/c_api.h>.
--
#if __GLASGOW_HASKELL__ >= 709
{-# LANGUAGE Safe #-}
#elif __GLASGOW_HASKELL__ >= 701
{-# LANGUAGE Trustworthy #-}
#endif
{-# LANGUAGE ForeignFunctionInterface #-}

module MXNet.Core.Internal.Raw where

import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Marshal.Utils ( with )
import Foreign.Ptr
import Foreign.Storable

import MXNet.Core.Internal.FFI
{#import MXNet.Core.Internal.Types.Raw #}

#include <mxnet/c_api.h>

-- | Handle size_t type.
{#typedef size_t CSize#}

{#fun MXGetLastError as mxGetLastError
    {
    } -> `String' #}

{#fun MXRandomSeed as mxRandomSeed
    { `Int'
    } -> `Int' #}

{#fun MXNotifyShutdown as mxNotifyShutdown
    {
    } -> `Int' #}

{#fun MXNDArrayCreateNone as mxNDArrayCreateNone
    { alloca- `NDArrayHandle' peek*
    } -> `Int' #}

{#fun MXNDArrayCreate as mxNDArrayCreate
    { with* `MXUInt'
    , id `MXUInt'
    , `Int'
    , `Int'
    , `Int'
    , alloca- `NDArrayHandle' peek*
    } -> `Int' #}

{#fun MXNDArrayCreateEx as mxNDArrayCreateEx
    { with* `MXUInt'
    , id `MXUInt'
    , `Int'
    , `Int'
    , `Int'
    , `Int'
    , alloca- `NDArrayHandle' peek*
    } -> `Int' #}

{#fun MXNDArrayLoadFromRawBytes as mxNDArrayLoadFromRawBytes
    { id `Ptr ()'
    , `CSize'
    , alloca- `NDArrayHandle' peek*
    } -> `Int' #}

{#fun MXNDArraySaveRawBytes as mxNDArraySaveRawBytes
    { id `NDArrayHandle'
    , alloca- `CSize' peek*
    , alloca- `Ptr CChar' peek* -- the head of returning memory bytes.
    } -> `Int' #}

{#fun MXNDArraySave as mxNDArraySave
    { `String'
    , id `MXUInt'
    , withArray* `[NDArrayHandle]'
    , withArray* `[Ptr CChar]' -- an array of string to C function.
    } -> `Int' #}

{#fun MXNDArrayLoad as mxNDArrayLoad
    { `String'
    , alloca- `MXUInt' peek*
    , id `Ptr (Ptr NDArrayHandle)' -- FIXME a pointer to store the address of
                                   -- the new array of string return from C function.
    , alloca- `MXUInt' peek*
    , id `Ptr (Ptr (Ptr CChar))' -- FIXME a pointer to store the address of
                                 -- the new array of string return from C function.
    } -> `Int' #}

{#fun MXNDArraySyncCopyFromCPU as mxNDArraySyncCopyFromCPU
    { id `NDArrayHandle'
    , id `Ptr ()' -- TODO better way out ?
    , `CSize'
    } -> `Int' #}

{#fun MXNDArraySyncCopyToCPU as mxNDArraySyncCopyToCPU
    { id `NDArrayHandle'
    , id `Ptr ()' -- TODO better way out ?
    , `CSize'
    } -> `Int' #}

{#fun MXNDArrayWaitToRead as mxNDArrayWaitToRead
    { id `NDArrayHandle'
    } -> `Int' #}

{#fun MXNDArrayWaitToWrite as mxNDArrayWaitToWrite
    { id `NDArrayHandle'
    } -> `Int' #}

{#fun MXNDArrayWaitAll as mxNDArrayWaitAll
    {
    } -> `Int' #}

{#fun MXNDArrayFree as mxNDArrayFree
    { id `NDArrayHandle'
    } -> `Int' #}

{#fun MXNDArraySlice as mxNDArraySlice
    { id `NDArrayHandle'
    , id `MXUInt'
    , id `MXUInt'
    , alloca- `NDArrayHandle' peek*
    } -> `Int' #}

{#fun MXNDArrayAt as mxNDArrayAt
    { id `NDArrayHandle'
    , id `MXUInt'
    , alloca- `NDArrayHandle' peek*
    } -> `Int' #}

{#fun MXNDArrayReshape as mxNDArrayReshape
    { id `NDArrayHandle'
    , `Int'
    , alloca- `Int' peekIntegral*
    , alloca- `NDArrayHandle' peek*
    } -> `Int' #}

{#fun MXNDArrayGetShape as mxNDArrayGetShape
    { id `NDArrayHandle'
    , with* `MXUInt'
    , alloca- `Ptr MXUInt' peek*
    } -> `Int' #}

{#fun MXNDArrayGetData as mxNDArrayGetData
    { id `NDArrayHandle'
    , alloca- `Ptr MXFloat' peek*
    } -> `Int' #}

{#fun MXNDArrayGetDType as mxNDArrayGetDType
    { id `NDArrayHandle'
    , alloca- `Int' peekIntegral*
    } -> `Int' #}

{#fun MXNDArrayGetContext as mxNDArrayGetContext
    { id `NDArrayHandle'
    , alloca- `Int' peekIntegral* -- device type
    , alloca- `Int' peekIntegral* -- device id
    } -> `Int' #}

{#fun MXListFunctions as mxListFunctions
    { alloca- `Int' peekIntegral*
    , alloca- `Ptr FunctionHandle' peek*
    } -> `Int' #}

{#fun MXGetFunction as mxGetFunction
    { `String'
    , alloca- `FunctionHandle' peek*
    } -> `Int' #}

{#fun MXFuncGetInfo as mxFuncGetInfo
    { id `FunctionHandle'
    , alloca- `String' peekString*   -- returned name of the function.
    , alloca- `String' peekString*   -- returned description of the function.
    , alloca- `Int' peekIntegral*    -- number of arguments of the function.
    , alloca- `Ptr (Ptr CChar)' peek* -- names of arguments.
    , alloca- `Ptr (Ptr CChar)' peek* -- types of arguments.
    , alloca- `Ptr (Ptr CChar)' peek* -- descriptions of arguments.
    , alloca- `String' peekString*    -- return type of the function.
    } -> `Int' #}                    -- FIXME representation of array of string.

{#fun MXFuncDescribe as mxFuncDescribe
    { id `FunctionHandle'
    , alloca- `MXUInt' peek*
    , alloca- `MXUInt' peek*
    , alloca- `MXUInt' peek*
    , alloca- `Int' peekIntegral*
    } -> `Int' #}

{#fun MXFuncInvoke as mxFuncInvoke
    { id `FunctionHandle'
    , withArray* `[NDArrayHandle]'
    , withArray* `[MXFloat]'
    , withArray* `[NDArrayHandle]'
    } -> `Int' #}

{#fun MXFuncInvokeEx as mxFuncInvokeEx
    { id `FunctionHandle'
    , withArray* `[NDArrayHandle]'
    , withArray* `[MXFloat]'
    , withArray* `[NDArrayHandle]'
    , `Int'
    , withArray* `[Ptr CChar]' -- TODO butter way out for two-dimension array arguments.
    , withArray* `[Ptr CChar]'
    } -> `Int' #}

{#fun MXSymbolListAtomicSymbolCreators as mxSymbolListAtomicSymbolCreators
    { alloca- `Int' peekIntegral*
    , alloca- `Ptr AtomicSymbolCreator' peek*
    } -> `Int' #}

{#fun MXSymbolGetAtomicSymbolName as mxSymbolGetAtomicSymbolName
    { id `AtomicSymbolCreator'
    , alloca- `String' peekString*
    } -> `Int' #}
