-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.Internal.TH
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Template haskell tools for finding Ops on NDArray and Symbol from dynamic library.
--
module MXNet.Core.Base.Internal.TH where

import Control.Monad
import Data.Char
import Language.Haskell.TH

import MXNet.Core.Internal.Types.Raw
import MXNet.Core.Base.Internal.Raw

-------------------------------------------------------------------------------

registerNDArrayOps :: Q [Dec]
registerNDArrayOps = runIO $ do
    names <- filter (isLower . head) <$> getNames
    -- print names
    return $ map (\name -> FunD (mkName name) [register name]) names

  where
    getNames :: IO [String]
    getNames = mxListAllOpNames >>= \(_, _, names) -> return names

    register fname =
            Clause [{- LitP . StringL $ fname -} ]
                   (NormalB . LitE $ StringL fname)
                   []

-------------------------------------------------------------------------------

registerSymbolOps :: Q [Dec]
registerSymbolOps = return []

-------------------------------------------------------------------------------

