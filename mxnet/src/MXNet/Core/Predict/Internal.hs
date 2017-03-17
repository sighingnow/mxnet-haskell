-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Predict.Internal
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Predict interfaces in core module of MXNet.
--
module MXNet.Core.Predict.Internal
    ( -- * Data types
      PredictorHandle
    , NDListHandle
      -- * Re-exports functions.
    , mxGetLastError
    , mxPredCreate
    , mxPredCreatePartialOut
    , mxPredGetOutputShape
    , mxPredSetInput
    , mxPredForward
    , mxPredPartialForward
    , mxPredGetOutput
    , mxPredFree
    , mxNDListCreate
    , mxNDListGet
    , mxNDListFree
    ) where

import MXNet.Core.Types.Internal ( PredictorHandle, NDListHandle )
import MXNet.Core.Base.Internal ( mxGetLastError )
import MXNet.Core.Predict.Internal.Raw

