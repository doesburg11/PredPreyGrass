using System.Collections.Generic;

using UnityEditor.IMGUI.Controls;
using UnityEngine;

namespace Unity.PlasticSCM.Editor.UI.Tree
{
    internal class PlasticTreeView : TreeView
    {
        internal PlasticTreeView() : base(new TreeViewState())
        {
            rowHeight = UnityConstants.TREEVIEW_ROW_HEIGHT;
            treeViewRect = new Rect(0, 0, 0, rowHeight);
            showAlternatingRowBackgrounds = false;
        }

        public override IList<TreeViewItem> GetRows()
        {
            return mRows;
        }

        protected override TreeViewItem BuildRoot()
        {
            return new TreeViewItem(0, -1, string.Empty);
        }

        protected override void BeforeRowsGUI()
        {
            int firstRowVisible;
            int lastRowVisible;
            GetFirstAndLastVisibleRows(out firstRowVisible, out lastRowVisible);

            GUI.DrawTexture(new Rect(0,
                firstRowVisible * rowHeight,
                GetRowRect(0).width + 1000,
                (lastRowVisible * rowHeight) + 1000),
                Images.GetTreeviewBackgroundTexture());

            DrawTreeViewItem.InitializeStyles();
            base.BeforeRowsGUI();
        }

        protected List<TreeViewItem> mRows = new List<TreeViewItem>();
    }
}
