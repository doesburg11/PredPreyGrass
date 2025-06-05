using System.Collections.Generic;
using System.Linq;

using Codice.Client.Common;
using Codice.LogWrapper;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.AssetUtils.Processor
{
    class AssetPostprocessor : UnityEditor.AssetPostprocessor
    {
        internal struct PathToMove
        {
            internal readonly string SrcPath;
            internal readonly string DstPath;

            internal PathToMove(string srcPath, string dstPath)
            {
                SrcPath = srcPath;
                DstPath = dstPath;
            }
        }

        internal static bool IsAutomaticAddEnabled { get; private set; }

        static AssetPostprocessor()
        {
            IsAutomaticAddEnabled = BoolSetting.Load(
                UnityConstants.AUTOMATIC_ADD_KEY_NAME, true);
        }

        internal static void Enable(
            string wkPath,
            PlasticAssetsProcessor plasticAssetsProcessor)
        {
            mLog.Debug("Enable");

            mWkPath = wkPath;
            mPlasticAssetsProcessor = plasticAssetsProcessor;

            mIsEnabled = true;
        }

        internal static void Disable()
        {
            mLog.Debug("Disable");

            mIsEnabled = false;

            mWkPath = null;
            mPlasticAssetsProcessor = null;
        }

        internal static void SetAutomaticAddPreference(bool isEnabled)
        {
            if (IsAutomaticAddEnabled == isEnabled)
                return;

            IsAutomaticAddEnabled = isEnabled;

            BoolSetting.Save(isEnabled, UnityConstants.AUTOMATIC_ADD_KEY_NAME);
        }

        internal static void SetIsRepaintNeededAfterAssetDatabaseRefresh()
        {
            mIsRepaintNeededAfterAssetDatabaseRefresh = true;
        }

        static void OnPostprocessAllAssets(
            string[] importedAssets,
            string[] deletedAssets,
            string[] movedAssets,
            string[] movedFromAssetPaths)
        {
            if (!mIsEnabled)
                return;

            if (mIsRepaintNeededAfterAssetDatabaseRefresh)
            {
                mIsRepaintNeededAfterAssetDatabaseRefresh = false;

                ProjectWindow.Repaint();
                RepaintInspector.All();
            }

            // Ensure that the MonoFSWatcher is enabled before processing Plastic operations.
            // It fixes the following scenario:
            // 1. Close PlasticSCM window.
            // 2. Create an asset, it appears with the added overlay.
            // 3. Open PlasticSCM window, the asset should appear as added instead of deleted locally.
            PlasticApp.EnableMonoFsWatcherIfNeeded();

            List<PathToMove> pathsToMove = GetPathsToMoveOnWorkspace(
                mWkPath, movedAssets, movedFromAssetPaths);

            mPlasticAssetsProcessor.MoveOnSourceControl(pathsToMove);

            List<string> pathsToCheckout;
            List<string> pathsToAdd;
            GetPathsToCheckoutOrAddOnWorkspace(
                mWkPath, importedAssets, pathsToMove, IsAutomaticAddEnabled,
                out pathsToCheckout, out pathsToAdd);

            mPlasticAssetsProcessor.CheckoutOnSourceControl(pathsToCheckout);

            mPlasticAssetsProcessor.DeleteFromSourceControl(
                GetPathsContainedOnWorkspace(mWkPath, deletedAssets));

            mPlasticAssetsProcessor.AddToSourceControl(pathsToAdd);

            // Unity Editor notifies the modified assets through
            // AssetModificationProcessor.OnWillSaveAssets before getting here.
            // Known Issue: renamed prefabs not triggering OnWillSaveAssets.
            mPlasticAssetsProcessor.CheckoutOnSourceControl(
                GetPathsContainedOnWorkspace(
                    mWkPath, AssetModificationProcessor.ExtractModifiedAssetsToProcess()));
        }

        static List<PathToMove> GetPathsToMoveOnWorkspace(
            string wkPath,
            string[] movedAssets,
            string[] movedFromAssetPaths)
        {
            List<PathToMove> proposedPathsToMove = GetPathsToMoveContainedOnWorkspace(
                wkPath, movedAssets, movedFromAssetPaths);

            // Unity Editor does not notify the moved assets in order. So, we have
            // to process the move operations in hierarchical order, otherwise, that
            // will end up with locally moved paths. This also avoids unnecessary
            // move operations for children when their parents are also moved.

            proposedPathsToMove.Sort((x, y) => PathHelper.
                GetPathMatchSorter().Compare(x.SrcPath, y.SrcPath));

            List<PathToMove> result = new List<PathToMove>();

            foreach (PathToMove proposedPathToMove in proposedPathsToMove)
            {
                if (result.Any(pathToMove => PathHelper.
                        IsContainedOn(proposedPathToMove.SrcPath, pathToMove.SrcPath)))
                    continue;

                result.Add(proposedPathToMove);
            }

            return result;
        }

        static void GetPathsToCheckoutOrAddOnWorkspace(
            string wkPath,
            string[] importedAssets,
            List<PathToMove> pathsToMove,
            bool isAutomaticAddEnabled,
            out List<string> pathsToCheckout,
            out List<string> pathsToAdd)
        {
            pathsToCheckout = new List<string>();
            pathsToAdd = new List<string>();

            // Unity Editor notifies as imported assets a combination of moved+modified
            // assets and added assets. To ensure proper version control operations,
            // we need to categorize them accordingly:
            // • moved+modified assets → check-out them to handle as controlled changes.
            // • added assets → add them when the automatic add preference is enabled.

            foreach (string asset in importedAssets)
            {
                string fullPath = AssetsPath.GetFullPathUnderWorkspace.
                    ForAsset(wkPath, asset);

                if (fullPath == null)
                    continue;

                if (pathsToMove.FindIndex(pathToMove =>
                        pathToMove.DstPath.Equals(fullPath)) != -1)
                {
                    pathsToCheckout.Add(fullPath);
                    continue;
                }

                if (isAutomaticAddEnabled)
                    pathsToAdd.Add(fullPath);
            }
        }

        static List<PathToMove> GetPathsToMoveContainedOnWorkspace(
            string wkPath,
            string[] movedAssets,
            string[] movedFromAssetPaths)
        {
            List<PathToMove> result = new List<PathToMove>(movedAssets.Length);

            for (int i = 0; i < movedAssets.Length; i++)
            {
                string fullSrcPath = AssetsPath.GetFullPathUnderWorkspace.
                    ForAsset(wkPath, movedFromAssetPaths[i]);

                if (fullSrcPath == null)
                    continue;

                string fullDstPath = AssetsPath.GetFullPathUnderWorkspace.
                    ForAsset(wkPath, movedAssets[i]);

                if (fullDstPath == null)
                    continue;

                result.Add(new PathToMove(
                    fullSrcPath, fullDstPath));
            }

            return result;
        }

        static List<string> GetPathsContainedOnWorkspace(
            string wkPath, string[] assets)
        {
            List<string> result = new List<string>(
                assets.Length);

            foreach (string asset in assets)
            {
                string fullPath = AssetsPath.GetFullPathUnderWorkspace.
                    ForAsset(wkPath, asset);

                if (fullPath == null)
                    continue;

                result.Add(fullPath);
            }

            return result;
        }

        static bool mIsEnabled;
        static bool mIsRepaintNeededAfterAssetDatabaseRefresh;

        static PlasticAssetsProcessor mPlasticAssetsProcessor;
        static string mWkPath;

        static readonly ILog mLog = PlasticApp.GetLogger("AssetPostprocessor");
    }
}
