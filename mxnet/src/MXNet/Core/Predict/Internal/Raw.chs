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
import Foreign.Ptr
import Foreign.Storable

import MXNet.Core.Internal.FFI
{#import MXNet.Core.Internal.Types.Raw #}

#include <mxnet/c_predict_api.h>

{#fun MXPredCreate as mxPredCreate
    { `String'
    , id `Ptr ()'
    , `Int'
    , `Int'
    , `Int'
    , id `MXUInt'
    , withArray* `[Ptr CChar]'
    , withArray* `[MXUInt]'
    , withArray* `[MXUInt]'
    , alloca- `PredictorHandle' peek*
    } -> `Int' #}

{#fun MXPredCreatePartialOut as mxPredCreatePartialOut
    { `String'
    , id `Ptr ()'
    , `Int'
    , `Int'
    , `Int'
    , id `MXUInt'
    , withArray* `[Ptr CChar]'
    , withArray* `[MXUInt]'
    , withArray* `[MXUInt]'
    , id `MXUInt'
    , withArray* `[Ptr CChar]'
    , alloca- `PredictorHandle' peek*
    } -> `Int' #}

{#fun MXPredGetOutputShape as mxPredGetOutputShape
    { id `PredictorHandle'
    , id `MXUInt'
    , alloca- `Ptr MXUInt' peek*
    , alloca- `MXUInt' peek*
    } -> `Int' #}

{#fun MXPredSetInput as mxPredSetInput
    { id `PredictorHandle'
    , `String'
    , withArray* `[MXFloat]'
    , id `MXUInt'
    } -> `Int' #}

{#fun MXPredForward as mxPredForward
    { id `PredictorHandle'
    } -> `Int' #}

{#fun MXPredPartialForward as mxPredPartialForward
    { id `PredictorHandle'
    , `Int'
    , alloca- `Int' peekIntegral*
    } -> `Int' #}

{#fun MXPredGetOutput as mxPredGetOutput
    { id `PredictorHandle'
    , id `MXUInt'
    , withArray* `[MXFloat]'
    , id `MXUInt'
    } -> `Int' #}

{#fun MXPredFree as mxPredFree
    { id `PredictorHandle'
    } -> `Int' #}

{#fun MXNDListCreate as mxNDListCreate
    { id `Ptr CChar'
    , `Int'
    , alloca- `NDListHandle' peek*
    , alloca- `MXUInt' peek*
    } -> `Int' #}

{#fun MXNDListGet as mxNDListGet
    { id `NDListHandle'
    , id `MXUInt'
    , alloca- `String' peekString*
    , alloca- `Ptr MXFloat' peek*
    , alloca- `Ptr MXUInt' peek*
    , alloca- `MXUInt' peek*
    } -> `Int' #}

{#fun MXNDListFree as mxNDListFree
    { id `NDListHandle'
    } -> `Int' #}
