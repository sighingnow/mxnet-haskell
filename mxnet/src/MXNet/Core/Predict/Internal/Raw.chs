-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Predict.Internal.Raw
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Direct C FFI bindings for <mxnet/c_predict_api.h>.
--
{-# LANGUAGE ForeignFunctionInterface #-}

module MXNet.Core.Predict.Internal.Raw where

import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable

import C2HS.C.Extra.Marshal

{#import MXNet.Core.Types.Internal.Raw #}

#include <mxnet/c_predict_api.h>

-- | Create a predictor.
{#fun MXPredCreate as mxPredCreate
    { `String'                      -- ^ The JSON string of the symbol.
    , id `Ptr ()'                   -- ^ The in-memory raw bytes of parameter ndarray file.
    , `Int'                         -- ^ The size of parameter ndarray file.
    , `Int'                         -- ^ The device type, 1: cpu, 2:gpu.
    , `Int'                         -- ^ The device id of the predictor.
    , id `MXUInt'                   -- ^ Number of input nodes to the net.
    , withStringArray* `[String]'   -- ^ The name of input argument.
    , withArray* `[MXUInt]'
    , withArray* `[MXUInt]'
    , alloca- `PredictorHandle' peek*
    } -> `Int' -- ^ The created predictor handle.
    #}

-- | Create a predictor wich customized outputs.
{#fun MXPredCreatePartialOut as mxPredCreatePartialOut
    { `String'
    , id `Ptr ()'
    , `Int'
    , `Int'
    , `Int'
    , id `MXUInt'
    , withStringArray* `[String]'  -- ^ The names of input arguments.
    , withArray* `[MXUInt]'
    , withArray* `[MXUInt]'
    , id `MXUInt'                  -- ^ Number of output nodes to the net.
    , alloca- `String' peekString* -- ^ The name of output argument.
    , alloca- `PredictorHandle' peek*
    } -> `Int' -- ^ The name of output argument and created predictor handle.
    #}

{#fun MXPredGetOutputShape as mxPredGetOutputShapeImpl
    { id `PredictorHandle'
    , id `MXUInt'
    , alloca- `Ptr MXUInt' peek*
    , alloca- `MXUInt' peek*
    } -> `Int' #}

-- | Get the shape of output node.
mxPredGetOutputShape :: PredictorHandle             -- ^ The predictor handle.
                     -> MXUInt                      -- ^ The index of output node, set to 0
                                                    -- if there is only one output.
                     -> IO (Int, [MXUInt], MXUInt)  -- ^ Output dimension and the shape data.
mxPredGetOutputShape handle index = do
    (res, p, d) <- mxPredGetOutputShapeImpl handle index
    shapes <- peekArray (fromIntegral d) p
    return (res, shapes, d)

-- | Set the input data of predictor.
{#fun MXPredSetInput as mxPredSetInput
    { id `PredictorHandle'
    , `String'                  -- ^ The name of input node to set.
    , withArray* `[MXFloat]'    -- ^ The pointer to the data to be set.
    , id `MXUInt'               -- ^ The size of data array, used for safety check.
    } -> `Int' #}

-- | Run a forward pass to get the output.
{#fun MXPredForward as mxPredForward
    { id `PredictorHandle'
    } -> `Int' #}

-- | Run a interactive forward pass to get the output.
{#fun MXPredPartialForward as mxPredPartialForward
    { id `PredictorHandle'
    , `Int'                         -- ^ The current step to run forward on.
    , alloca- `Int' peekIntegral*
    } -> `Int' -- ^ The number of steps left.
    #}

-- | Get the output value of prediction.
{#fun MXPredGetOutput as mxPredGetOutput
    { id `PredictorHandle'
    , id `MXUInt'       -- ^ The index of output node, set to 0 if there is only one output.
    , id `Ptr MXFloat'  -- ^ __/User allocated/__ data to hold the output.
    , id `MXUInt'       -- ^ The size of data array, used for safe checking.
    } -> `Int' #}

-- | Free a predictor handle.
{#fun MXPredFree as mxPredFree
    { id `PredictorHandle'
    } -> `Int' #}

-- | Create a NDArray List by loading from ndarray file.
{#fun MXNDListCreate as mxNDListCreate
    { id `Ptr CChar'                -- ^ The byte contents of nd file to be loaded.
    , `Int'                         -- ^ The size of the nd file to be loaded.
    , alloca- `NDListHandle' peek*
    , alloca- `MXUInt' peek*
    } -> `Int' -- ^ The out put NDListHandle and length of the list.
    #}

{#fun MXNDListGet as mxNDListGetImpl
    { id `NDListHandle'
    , id `MXUInt'
    , alloca- `String' peekString*
    , alloca- `Ptr MXFloat' peek*
    , alloca- `Ptr MXUInt' peek*
    , alloca- `MXUInt' peek*
    } -> `Int' #}

-- | Get an element from list.
mxNDListGet :: NDListHandle
            -> MXUInt                                       -- ^ The index in the list.
            -> IO (Int,
                   String, Ptr MXFloat, [MXUInt], MXUInt)   -- ^ The name of output, the data
                                                            -- region of the item, the shape of
                                                            -- the item and shape's dimension.
mxNDListGet handle index = do
    (res, name, output, p, d) <- mxNDListGetImpl handle index
    shapes <- peekArray (fromIntegral d) p
    return (res, name, output, shapes, d)

-- | Free a predictor handle.
{#fun MXNDListFree as mxNDListFree
    { id `NDListHandle'
    } -> `Int' #}
