-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Types.Internal.Raw
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Collect data type defintions into a single raw binding module to avoid redefinitions.
--
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE DeriveGeneric #-}

module MXNet.Core.Types.Internal.Raw where

import Foreign.C.Types
import Foreign.Ptr
import Foreign.Storable
import Foreign.ForeignPtr
import Foreign.ForeignPtr.Unsafe
import Foreign.Marshal.Array (withArray)
import GHC.Generics
import Control.Monad ((>=>))

#include <nnvm/c_api.h>
#include <mxnet/c_api.h>
#include <mxnet/c_predict_api.h>

-- | Handle size_t type.
{#typedef size_t CSize#}

{---------------------------------------------------------------------
- Primitive type alias.
---------------------------------------------------------------------}

-- | NNUint type alias.
type NNUInt = CUInt

-- | MXUint type alias.
type MXUInt = CUInt

-- | MXFloat type alias.
type MXFloat = CFloat

{---------------------------------------------------------------------
- <nnvm/c_api.h>
---------------------------------------------------------------------}

-- | Handle to a function that takes param and creates symbol.

{#pointer OpHandle #}

{- FIXME maybe a bug from c2hs, when make a type alias of OpHandle, the
   generated CFFI function will not be correct.

{#pointer OpHandle newtype #}

instance Storable OpHandle where
    sizeOf (OpHandle t) = sizeOf t
    alignment (OpHandle t) = alignment t
    peek p = fmap OpHandle (peek (castPtr p))
    poke p (OpHandle t) = poke (castPtr p) t

--}

-- | Handle to a symbol that can be bind as operator.
{#pointer SymbolHandle newtype #}
deriving instance Generic SymbolHandle

instance Storable SymbolHandle where
    sizeOf (SymbolHandle t) = sizeOf t
    alignment (SymbolHandle t) = alignment t
    peek p = fmap SymbolHandle (peek (castPtr p))
    poke p (SymbolHandle t) = poke (castPtr p) t

-- | Handle to Graph.
{#pointer GraphHandle newtype #}
deriving instance Generic GraphHandle

instance Storable GraphHandle where
    sizeOf (GraphHandle t) = sizeOf t
    alignment (GraphHandle t) = alignment t
    peek p = fmap GraphHandle (peek (castPtr p))
    poke p (GraphHandle t) = poke (castPtr p) t

{---------------------------------------------------------------------
- <mxnet/c_api.h>
---------------------------------------------------------------------}

-- | Handle to NDArray.
{#pointer NDArrayHandle foreign finalizer MXNDArrayFree as mxNDArrayFree newtype #}
deriving instance Generic NDArrayHandle
type NDArrayHandlePtr = Ptr NDArrayHandle

newNDArrayHandle :: NDArrayHandlePtr -> IO NDArrayHandle
newNDArrayHandle = newForeignPtr mxNDArrayFree >=> return . NDArrayHandle

peekNDArrayHandle :: Ptr NDArrayHandlePtr -> IO NDArrayHandle
peekNDArrayHandle = peek >=> newNDArrayHandle

withNDArrayHandleArray :: [NDArrayHandle] -> (Ptr NDArrayHandlePtr -> IO r) -> IO r
withNDArrayHandleArray array io = do
    let unNDArrayHandle (NDArrayHandle fptr) = fptr
    r <- withArray (map (unsafeForeignPtrToPtr . unNDArrayHandle) array) io
    mapM_ (touchForeignPtr . unNDArrayHandle) array
    return r

-- | Handle to a mxnet narray function that changes NDArray.
{#pointer FunctionHandle newtype #}
deriving instance Generic FunctionHandle

instance Storable FunctionHandle where
    sizeOf (FunctionHandle t) = sizeOf t
    alignment (FunctionHandle t) = alignment t
    peek p = fmap FunctionHandle (peek (castPtr p))
    poke p (FunctionHandle t) = poke (castPtr p) t

-- | Handle to a function that takes param and creates symbol.
type AtomicSymbolCreator = OpHandle
deriving instance Generic AtomicSymbolHandle

-- | Handle to a AtomicSymbol.
{#pointer AtomicSymbolHandle newtype #}

instance Storable AtomicSymbolHandle where
    sizeOf (AtomicSymbolHandle t) = sizeOf t
    alignment (AtomicSymbolHandle t) = alignment t
    peek p = fmap AtomicSymbolHandle (peek (castPtr p))
    poke p (AtomicSymbolHandle t) = poke (castPtr p) t

{#pointer ExecutorHandle foreign finalizer MXExecutorFree as mxExecutorFree newtype #}
deriving instance Generic ExecutorHandle

type ExecutorHandlePtr = Ptr ExecutorHandle

newExecutorHandle :: ExecutorHandlePtr -> IO ExecutorHandle
newExecutorHandle = newForeignPtr mxExecutorFree >=> return . ExecutorHandle

peekExecutorHandle :: Ptr ExecutorHandlePtr -> IO ExecutorHandle
peekExecutorHandle = peek >=> newExecutorHandle

-- | Handle a dataiter creator.
{#pointer DataIterCreator newtype #}
deriving instance Generic DataIterCreator

instance Storable DataIterCreator where
    sizeOf (DataIterCreator t) = sizeOf t
    alignment (DataIterCreator t) = alignment t
    peek p = fmap DataIterCreator (peek (castPtr p))
    poke p (DataIterCreator t) = poke (castPtr p) t

-- | Handle to a DataIterator.
{#pointer DataIterHandle newtype #}
deriving instance Generic DataIterHandle

instance Storable DataIterHandle where
    sizeOf (DataIterHandle t) = sizeOf t
    alignment (DataIterHandle t) = alignment t
    peek p = fmap DataIterHandle (peek (castPtr p))
    poke p (DataIterHandle t) = poke (castPtr p) t

-- | Handle to KVStore.
{#pointer KVStoreHandle newtype #}
deriving instance Generic KVStoreHandle

instance Storable KVStoreHandle where
    sizeOf (KVStoreHandle t) = sizeOf t
    alignment (KVStoreHandle t) = alignment t
    peek p = fmap KVStoreHandle (peek (castPtr p))
    poke p (KVStoreHandle t) = poke (castPtr p) t

-- | Handle to RecordIO.
{#pointer RecordIOHandle newtype #}
deriving instance Generic RecordIOHandle

instance Storable RecordIOHandle where
    sizeOf (RecordIOHandle t) = sizeOf t
    alignment (RecordIOHandle t) = alignment t
    peek p = fmap RecordIOHandle (peek (castPtr p))
    poke p (RecordIOHandle t) = poke (castPtr p) t

-- | Handle to MXRtc.
{#pointer RtcHandle newtype #}
deriving instance Generic RtcHandle

instance Storable RtcHandle where
    sizeOf (RtcHandle t) = sizeOf t
    alignment (RtcHandle t) = alignment t
    peek p = fmap RtcHandle (peek (castPtr p))
    poke p (RtcHandle t) = poke (castPtr p) t

-- | Callback: ExecutorMonitorCallback.
{#pointer ExecutorMonitorCallback newtype #}
deriving instance Generic ExecutorMonitorCallback

-- | Callback: CustomOpPropCreator.
{#pointer CustomOpPropCreator newtype #}
deriving instance Generic CustomOpPropCreator

-- | Callback: MXKVStoreUpdater, user-defined updater for the kvstore.
type MXKVStoreUpdater = Int             -- ^ The key.
                      -> NDArrayHandlePtr  -- ^ The pushed value on the key.
                      -> NDArrayHandlePtr  -- ^ The value stored on local on the key.
                      -> Ptr ()         -- ^ The additional handle to the updater.
                      -> IO Int

foreign import ccall "wrapper"
    makeMXKVStoreUpdater :: MXKVStoreUpdater -> IO (FunPtr MXKVStoreUpdater)

-- | Callback: MXKVStoreServerController, the prototype of a server controller.
type MXKVStoreServerController = Int        -- ^ The head of the command.
                               -> Ptr CChar -- ^ The body of the command.
                               -> Ptr ()    -- ^ Helper handle for implementing controller.
                               -> IO Int

foreign import ccall "wrapper"
    makeMXKVStoreServerController :: MXKVStoreServerController -> IO (FunPtr MXKVStoreServerController)

{---------------------------------------------------------------------
- <mxnet/c_predict_api.h>
---------------------------------------------------------------------}

-- | Handle to Predictor.
{#pointer PredictorHandle newtype #}

instance Storable PredictorHandle where
    sizeOf (PredictorHandle t) = sizeOf t
    alignment (PredictorHandle t) = alignment t
    peek p = fmap PredictorHandle (peek (castPtr p))
    poke p (PredictorHandle t) = poke (castPtr p) t

-- | Handle to NDArrayList.
{#pointer NDListHandle newtype #}

instance Storable NDListHandle where
    sizeOf (NDListHandle t) = sizeOf t
    alignment (NDListHandle t) = alignment t
    peek p = fmap NDListHandle (peek (castPtr p))
    poke p (NDListHandle t) = poke (castPtr p) t
