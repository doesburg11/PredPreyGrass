using UnityEditor.IMGUI.Controls;

using PlasticGui;
using Unity.PlasticSCM.Editor.Views.PendingChanges;

namespace Unity.PlasticSCM.Editor.UI.Tree
{
    internal static class TreeViewItemExtensions
    {
        internal static IPlasticTreeNode GetPlasticTreeNode(this TreeViewItem item)
        {
            if (item is ChangelistTreeViewItem)
            {
                return ((ChangelistTreeViewItem)item).Changelist;
            }

            if (item is ChangeCategoryTreeViewItem)
            {
                return ((ChangeCategoryTreeViewItem)item).Category;
            }

            return ((ChangeTreeViewItem)item).ChangeInfo;
        }
    }
}
