using System.Collections.Generic;
using System.Linq;

using Codice.Client.BaseCommands;
using Codice.CM.Common;
using Unity.PlasticSCM.Editor.AssetUtils;

namespace Unity.PlasticSCM.Editor
{
    internal static class ProjectPackages
    {
        internal static bool ShouldBeResolvedFromPaths(
            WorkspaceInfo wkInfo, List<string> updatedItems)
        {
            if (IsDynamicWorkspace(wkInfo))
                return true;

            return updatedItems.Any(ShouldPathBeResolved);
        }

        internal static bool ShouldBeResolvedFromUpdateReport(
            WorkspaceInfo wkInfo, List<string> updatedItems)
        {
            if (IsDynamicWorkspace(wkInfo))
                return true;

            updatedItems = updatedItems.Select(GetPathFromUpdateReport).ToList();

            return updatedItems.Any(ShouldPathBeResolved);
        }

        internal static bool ShouldBeResolvedFromUpdateProgress(
           WorkspaceInfo wkInfo, UpdateProgress progress)
        {
            if (progress == null)
                return false;

            if (IsDynamicWorkspace(wkInfo))
                return true;

            return ShouldBeResolved(progress.AddedItems.Where(i => !i.IsDirectory))
                || ShouldBeResolved(progress.DeletedItems)
                || ShouldBeResolved(progress.ChangedItems.Where(i => !i.IsDirectory))
                || ShouldBeResolved(progress.MovedItems);
        }

        static bool IsDynamicWorkspace(WorkspaceInfo wkInfo)
        {
            // We cannot obtain the updated items from a dynamic workspace, so for the moment,
            // we'll force the Packages reimport for these kind of workspaces.
            return Codice.CM.WorkspaceServer.IsDynamicWorkspace.Check(wkInfo);
        }

        static bool ShouldBeResolved(IEnumerable<UpdateProgress.UpdatedItem> items)
        {
            return items.Select(i => i.Path).Any(ShouldPathBeResolved)
                || items.Any(i => i.IsDirectory);
        }

        static bool ShouldBeResolved(IEnumerable<UpdateProgress.UpdatedMovedItem> items)
        {
            return items.Select(i => i.DstPath).Any(ShouldPathBeResolved)
                || items.Any(i => i.IsDirectory);
        }

        static bool ShouldPathBeResolved(string path)
        {
            return AssetsPath.IsPackagesRootElement(path)
                || AssetsPath.IsScript(path);
        }

        static string GetPathFromUpdateReport(string item)
        {
            if (string.IsNullOrEmpty(item))
                return string.Empty;

            // For full workspaces we expect to receive the updated items with format <{UPDATE_TYPE}:{ITEM_PATH}>
            if (!item.StartsWith("<") || !item.EndsWith(">"))
                return string.Empty;

            int startIndex = item.IndexOf(":") + 1;

            if (startIndex == 0)
                return string.Empty;

            int endIndex = item.Length - 1;

            return item.Substring(startIndex, endIndex - startIndex);
        }
    }
}
