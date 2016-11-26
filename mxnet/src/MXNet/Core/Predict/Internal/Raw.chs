-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Predict.Internal.Raw
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Direct C FFI bindings for <mxnet/c_predict_api.h>.
--
#if __GLASGOW_HASKELL__ >= 709
{-# LANGUAGE Safe #-}
#elif __GLASGOW_HASKELL__ >= 701
{-# LANGUAGE Trustworthy #-}
#endif
{-# LANGUAGE ForeignFunctionInterface #-}

module MXNet.Core.Predict.Internal.Raw where

import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Marshal.Utils ( with )
import Foreign.Ptr
import Foreign.Storable

import MXNet.Core.Internal.FFI

#include <mxnet/c_predict_api.h>
