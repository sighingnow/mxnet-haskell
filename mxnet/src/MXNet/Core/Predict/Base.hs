-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Predict.Base
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Predict interfaces in core module of MXNet.
--
module MXNet.Core.Predict.Base (
      -- * Data types
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

import MXNet.Core.Internal.Types.Raw ( PredictorHandle, NDListHandle )
import MXNet.Core.Base.Internal.Raw ( mxGetLastError )
import MXNet.Core.Predict.Internal.Raw
