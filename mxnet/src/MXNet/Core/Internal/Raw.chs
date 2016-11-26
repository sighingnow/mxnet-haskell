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

#include <mxnet/c_api.h>

-- | Handle size_t type.
{#typedef size_t CSize#}

-- | MXUint type alias.
type MXUint = CUInt

-- | MXFloat type alias.
type MXFloat = CFloat

-- | Handle to NDArray.
{#pointer NDArrayHandle newtype #}

instance Storable NDArrayHandle where
    sizeOf (NDArrayHandle t) = sizeOf t
    alignment (NDArrayHandle t) = alignment t
    peek p = fmap NDArrayHandle (peek (castPtr p))
    poke p (NDArrayHandle t) = poke (castPtr p) t

-- | Handle to a mxnet narray function that changes NDArray.
{#pointer FunctionHandle newtype #}

instance Storable FunctionHandle where
    sizeOf (FunctionHandle t) = sizeOf t
    alignment (FunctionHandle t) = alignment t
    peek p = fmap FunctionHandle (peek (castPtr p))
    poke p (FunctionHandle t) = poke (castPtr p) t

-- | Handle to a function that takes param and creates symbol.
{#pointer AtomicSymbolCreator newtype #}

instance Storable AtomicSymbolCreator where
    sizeOf (AtomicSymbolCreator t) = sizeOf t
    alignment (AtomicSymbolCreator t) = alignment t
    peek p = fmap AtomicSymbolCreator (peek (castPtr p))
    poke p (AtomicSymbolCreator t) = poke (castPtr p) t

-- | Handle to a symbol that can be bind as operator.
{#pointer SymbolHandle newtype #}

instance Storable SymbolHandle where
    sizeOf (SymbolHandle t) = sizeOf t
    alignment (SymbolHandle t) = alignment t
    peek p = fmap SymbolHandle (peek (castPtr p))
    poke p (SymbolHandle t) = poke (castPtr p) t

-- | Handle to a AtomicSymbol.
{#pointer AtomicSymbolHandle newtype #}

instance Storable AtomicSymbolHandle where
    sizeOf (AtomicSymbolHandle t) = sizeOf t
    alignment (AtomicSymbolHandle t) = alignment t
    peek p = fmap AtomicSymbolHandle (peek (castPtr p))
    poke p (AtomicSymbolHandle t) = poke (castPtr p) t

{#pointer ExecutorHandle newtype #}

-- | Handle to an Executor.
instance Storable ExecutorHandle where
    sizeOf (ExecutorHandle t) = sizeOf t
    alignment (ExecutorHandle t) = alignment t
    peek p = fmap ExecutorHandle (peek (castPtr p))
    poke p (ExecutorHandle t) = poke (castPtr p) t

-- | Handle a dataiter creator.
{#pointer DataIterCreator newtype #}

instance Storable DataIterCreator where
    sizeOf (DataIterCreator t) = sizeOf t
    alignment (DataIterCreator t) = alignment t
    peek p = fmap DataIterCreator (peek (castPtr p))
    poke p (DataIterCreator t) = poke (castPtr p) t

-- | Handle to a DataIterator.
{#pointer DataIterHandle newtype #}

instance Storable DataIterHandle where
    sizeOf (DataIterHandle t) = sizeOf t
    alignment (DataIterHandle t) = alignment t
    peek p = fmap DataIterHandle (peek (castPtr p))
    poke p (DataIterHandle t) = poke (castPtr p) t

-- | Handle to KVStore.
{#pointer KVStoreHandle newtype #}

instance Storable KVStoreHandle where
    sizeOf (KVStoreHandle t) = sizeOf t
    alignment (KVStoreHandle t) = alignment t
    peek p = fmap KVStoreHandle (peek (castPtr p))
    poke p (KVStoreHandle t) = poke (castPtr p) t

-- | Handle to RecordIO.
{#pointer RecordIOHandle newtype #}

instance Storable RecordIOHandle where
    sizeOf (RecordIOHandle t) = sizeOf t
    alignment (RecordIOHandle t) = alignment t
    peek p = fmap RecordIOHandle (peek (castPtr p))
    poke p (RecordIOHandle t) = poke (castPtr p) t

-- | Handle to MXRtc.
{#pointer RtcHandle newtype #}

instance Storable RtcHandle where
    sizeOf (RtcHandle t) = sizeOf t
    alignment (RtcHandle t) = alignment t
    peek p = fmap RtcHandle (peek (castPtr p))
    poke p (RtcHandle t) = poke (castPtr p) t

-- | Handle to a function that takes param and creates optimizer.
{#pointer OptimizerCreator newtype #}

instance Storable OptimizerCreator where
    sizeOf (OptimizerCreator t) = sizeOf t
    alignment (OptimizerCreator t) = alignment t
    peek p = fmap OptimizerCreator (peek (castPtr p))
    poke p (OptimizerCreator t) = poke (castPtr p) t

-- | Handle to Optimizer.
{#pointer OptimizerHandle newtype #}

instance Storable OptimizerHandle where
    sizeOf (OptimizerHandle t) = sizeOf t
    alignment (OptimizerHandle t) = alignment t
    peek p = fmap OptimizerHandle (peek (castPtr p))
    poke p (OptimizerHandle t) = poke (castPtr p) t

-- | Callback: ExecutorMonitorCallback.
{#pointer ExecutorMonitorCallback newtype #}

-- | Callback: CustomOpPropCreator.
{#pointer CustomOpPropCreator newtype #}

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
    { with* `MXUint'
    , id `MXUint'
    , `Int'
    , `Int'
    , `Int'
    , alloca- `NDArrayHandle' peek*
    } -> `Int' #}

{#fun MXNDArrayCreateEx as mxNDArrayCreateEx
    { with* `MXUint'
    , id `MXUint'
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
    , id `MXUint'
    , withArray* `[NDArrayHandle]'
    , withArray* `[Ptr CChar]' -- an array of string to C function.
    } -> `Int' #}

{#fun MXNDArrayLoad as mxNDArrayLoad
    { `String'
    , alloca- `MXUint' peek*
    , id `Ptr (Ptr NDArrayHandle)' -- FIXME a pointer to store the address of
                                   -- the new array of string return from C function.
    , alloca- `MXUint' peek*
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
    , id `MXUint'
    , id `MXUint'
    , alloca- `NDArrayHandle' peek*
    } -> `Int' #}

{#fun MXNDArrayAt as mxNDArrayAt
    { id `NDArrayHandle'
    , id `MXUint'
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
    , with* `MXUint'
    , alloca- `Ptr MXUint' peek*
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
    , alloca- `MXUint' peek*
    , alloca- `MXUint' peek*
    , alloca- `MXUint' peek*
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
