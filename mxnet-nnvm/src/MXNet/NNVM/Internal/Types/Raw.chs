-----------------------------------------------------------
-- |
-- module:                      MXNet.NNVM.Internal.Types.Raw
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Collect data type defintions of NNVM into a single raw binding module to
-- avoid redefinitions.
--
#if __GLASGOW_HASKELL__ >= 709
{-# LANGUAGE Safe #-}
#elif __GLASGOW_HASKELL__ >= 701
{-# LANGUAGE Trustworthy #-}
#endif
{-# LANGUAGE ForeignFunctionInterface #-}

module MXNet.NNVM.Internal.Types.Raw where

import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Marshal.Utils ( with )
import Foreign.Ptr
import Foreign.Storable

#include <nnvm/c_api.h>

{---------------------------------------------------------------------
- <nnvm/c_api.h>
---------------------------------------------------------------------}

-- | MXUint type alias.
type NNUInt = CUInt

-- | Handle to a function that takes param and creates symbol.
{#pointer OpHandle newtype #}

instance Storable OpHandle where
    sizeOf (OpHandle t) = sizeOf t
    alignment (OpHandle t) = alignment t
    peek p = fmap OpHandle (peek (castPtr p))
    poke p (OpHandle t) = poke (castPtr p) t

-- | Handle to a symbol that can be bind as operator.
{#pointer SymbolHandle newtype #}

instance Storable SymbolHandle where
    sizeOf (SymbolHandle t) = sizeOf t
    alignment (SymbolHandle t) = alignment t
    peek p = fmap SymbolHandle (peek (castPtr p))
    poke p (SymbolHandle t) = poke (castPtr p) t

-- | Handle to Graph.
{#pointer GraphHandle newtype #}

instance Storable GraphHandle where
    sizeOf (GraphHandle t) = sizeOf t
    alignment (GraphHandle t) = alignment t
    peek p = fmap GraphHandle (peek (castPtr p))
    poke p (GraphHandle t) = poke (castPtr p) t
