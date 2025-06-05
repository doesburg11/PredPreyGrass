using System;
using System.Collections.Generic;
using System.Linq;

using UnityEditor;
using UnityEditor.IMGUI.Controls;

using PlasticGui;

namespace Unity.PlasticSCM.Editor.UI.Tree
{
    internal class TreeViewSessionState
    {
        internal static void Save(
            TreeView treeView,
            string uncheckedKey)
        {
            var rows = treeView.GetRows();

            if (!rows.Any())
                return;

            List<string> uncheckedRows = new List<string>();

            for (int i = 0; i < rows.Count; i++)
            {
                bool? isChecked = CheckableItems.GetIsCheckedValue(
                    rows[i].GetPlasticTreeNode());

                if (!isChecked.HasValue)
                    continue;

                if (string.IsNullOrEmpty(rows[i].displayName))
                    continue;

                if (!isChecked.Value)
                    uncheckedRows.Add(rows[i].displayName);
            }

            SessionState.SetString(uncheckedKey, string.Join(":", uncheckedRows));
        }

        internal static void Restore(
            TreeView treeView,
            string uncheckedKey)
        {
            var rows = treeView.GetRows();

            if (!rows.Any())
                return;

            string uncheckedRows = SessionState.GetString(
                uncheckedKey, string.Empty);

            if (string.IsNullOrEmpty(uncheckedRows))
                return;

            string[] uncheckedArray = uncheckedRows.Split(':');

            for (int i = 0; i < rows.Count; i++)
            {
                if (string.IsNullOrEmpty(rows[i].displayName))
                    continue;

                if (uncheckedArray.Contains(rows[i].displayName))
                {
                    CheckableItems.SetCheckedValue(
                        rows[i].GetPlasticTreeNode(), false);
                    continue;
                }

                CheckableItems.SetCheckedValue(
                    rows[i].GetPlasticTreeNode(), true);
            }

            // Clear session state after the every update
            SessionState.EraseString(uncheckedKey);
        }
    }
}
