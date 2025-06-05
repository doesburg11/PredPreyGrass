using UnityEditor.IMGUI.Controls;

namespace Unity.PlasticSCM.Editor.Views.Shelves
{
    class ShelveListViewItem : TreeViewItem
    {
        internal object ObjectInfo { get; private set; }

        internal ShelveListViewItem(int id, object objectInfo)
            : base(id, 1)
        {
            ObjectInfo = objectInfo;

            displayName = id.ToString();
        }
    }
}
