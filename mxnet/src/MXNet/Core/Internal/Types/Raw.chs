-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Internal.Types.Raw
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Collect data type defintions into a single raw binding module to avoid redefinitions.
--
#if __GLASGOW_HASKELL__ >= 709
{-# LANGUAGE Safe #-}
#elif __GLASGOW_HASKELL__ >= 701
{-# LANGUAGE Trustworthy #-}
#endif
{-# LANGUAGE ForeignFunctionInterface #-}

module MXNet.Core.Internal.Types.Raw where

import Foreign.C.Types
import Foreign.Ptr
import Foreign.Storable

#include <mxnet/c_api.h>
#include <mxnet/c_predict_api.h>

-- | Handle size_t type.
{#typedef size_t CSize#}

{---------------------------------------------------------------------
- <mxnet/c_api.h>
---------------------------------------------------------------------}

-- | MXUint type alias.
type MXUInt = CUInt

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
