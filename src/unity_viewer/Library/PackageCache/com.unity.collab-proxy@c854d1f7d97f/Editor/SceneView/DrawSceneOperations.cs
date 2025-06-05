using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

using UnityEditor.VersionControl;

using Codice.Client.Common.Threading;
using Codice.CM.Common;
using Codice.LogWrapper;
using PlasticGui;
using Unity.PlasticSCM.Editor.AssetUtils;
using Unity.PlasticSCM.Editor.AssetsOverlays.Cache;
using Unity.PlasticSCM.Editor.AssetMenu;
using Unity.PlasticSCM.Editor.AssetUtils.Processor;

using PlasticAssetModificationProcessor = Unity.PlasticSCM.Editor.AssetUtils.Processor.AssetModificationProcessor;

namespace Unity.PlasticSCM.Editor.SceneView
{
    static class DrawSceneOperations
    {
        internal static void Enable(
            WorkspaceInfo wkInfo,
            IPlasticAPI plasticApi,
            IWorkspaceOperationsMonitor workspaceOperationsMonitor,
            IAssetStatusCache assetStatusCache)
        {
            if (mIsEnabled)
                return;

            mLog.Debug("Enable");

            mWkInfo = wkInfo;
            mPlasticAPI = plasticApi;
            mWorkspaceOperationsMonitor = workspaceOperationsMonitor;
            mAssetStatusCache = assetStatusCache;

            mIsEnabled = true;

            Provider.preCheckoutCallback += Provider_preCheckoutCallback;
        }

        internal static void Disable()
        {
            mLog.Debug("Disable");

            mIsEnabled = false;

            Provider.preCheckoutCallback -= Provider_preCheckoutCallback;

            mWkInfo = null;
            mPlasticAPI = null;
            mWorkspaceOperationsMonitor = null;
            mAssetStatusCache = null;
        }

        static bool Provider_preCheckoutCallback(
            AssetList list,
            ref string changesetID,
            ref string changesetDescription)
        {
            try
            {
                if (!mIsEnabled)
                    return true;

                List<Asset> assets = GetUnmodifiedAssets(
                    list, PlasticAssetModificationProcessor.GetModifiedAssetsToProcess());

                if (assets.Count == 0)
                    return true;

                List<string> selectedPaths = GetSelectedPaths.ForOperation(
                    mWkInfo, assets, mPlasticAPI, mAssetStatusCache,
                    AssetMenuOperations.Checkout);

                if (selectedPaths.Count == 0)
                    return true;

                mWorkspaceOperationsMonitor.AddPathsToCheckout(selectedPaths);
            }
            catch (Exception ex)
            {
                ExceptionsHandler.LogException(typeof(DrawSceneOperations).Name, ex);
            }

            return true;
        }

        static List<Asset> GetUnmodifiedAssets(
            AssetList assetList,
            ReadOnlyCollection<string> modifiedAssetsToProcess)
        {
            List<Asset> result = new List<Asset>(assetList.Count);

            foreach (Asset asset in assetList)
            {
                if (modifiedAssetsToProcess.Contains(asset.path))
                    continue;

                result.Add(asset);
            }

            return result;
        }

        static bool mIsEnabled;
        static IAssetStatusCache mAssetStatusCache;
        static IWorkspaceOperationsMonitor mWorkspaceOperationsMonitor;
        static IPlasticAPI mPlasticAPI;
        static WorkspaceInfo mWkInfo;

        static readonly ILog mLog = PlasticApp.GetLogger("DrawSceneOperations");
    }
}
