-----------------------------------------------------------
-- |
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Example of how to work with Symbol.
--
import           MXNet.Core.Base
import           MXNet.Core.Symbol

naive :: IO ()
naive = do
    (r, n, creators) <- mxSymbolListAtomicSymbolCreators
    print n
    mapM_ (\x -> mxSymbolGetAtomicSymbolName x >>= print) creators

main :: IO ()
main = do
    var <- makeVar "abde"
    getVarName var >>= print
