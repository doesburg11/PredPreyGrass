using System.Collections.Generic;

using Codice;
using Codice.Client.BaseCommands;
using Codice.Client.Commands.WkTree;
using PlasticGui.WorkspaceWindow;

namespace Unity.PlasticSCM.Editor.AssetsOverlays.Cache
{
    internal class LocalStatusCache
    {
        internal AssetStatus GetStatus(string fullPath)
        {
            AssetStatus result;

            if (mStatusByPathCache.TryGetValue(fullPath, out result))
                return result;

            result = CalculateStatus(
                fullPath,
                FilterManager.Get().GetIgnoredFilter(),
                FilterManager.Get().GetHiddenChangesFilter());

            mStatusByPathCache.Add(fullPath, result);

            return result;
        }

        internal void Clear()
        {
            mStatusByPathCache.Clear();
        }

        static AssetStatus CalculateStatus(
            string fullPath,
            IgnoredFilesFilter ignoredFilter,
            HiddenChangesFilesFilter hiddenChangesFilter)
        {
            WorkspaceTreeNode treeNode = PlasticGui.Plastic.API.GetWorkspaceTreeNode(fullPath);

            if (CheckWorkspaceTreeNodeStatus.IsPrivate(treeNode))
            {
                return ignoredFilter.IsIgnored(fullPath) ?
                    AssetStatus.Ignored : AssetStatus.Private;
            }

            if (CheckWorkspaceTreeNodeStatus.IsAdded(treeNode))
                return AssetStatus.Added;

            AssetStatus result = AssetStatus.Controlled;

            if (CheckWorkspaceTreeNodeStatus.IsCheckedOut(treeNode) &&
                !CheckWorkspaceTreeNodeStatus.IsDirectory(treeNode))
                result |= AssetStatus.Checkout;

            if (hiddenChangesFilter.IsHiddenChanged(fullPath))
                result |= AssetStatus.HiddenChanged;

            return result;
        }

        Dictionary<string, AssetStatus> mStatusByPathCache =
            BuildPathDictionary.ForPlatform<AssetStatus>();
    }
}
