using System.Collections.Generic;
using System.Linq;

using Codice.CM.Common;
using Unity.PlasticSCM.Editor.UI.Tree;

namespace Unity.PlasticSCM.Editor.Views.Shelves
{
    internal static class ShelvesSelection
    {
        internal static void SelectShelves(
            ShelvesListView listView,
            List<RepObjectInfo> shelvesToSelect,
            int defaultRow)
        {
            if (shelvesToSelect == null || shelvesToSelect.Count == 0)
            {
                TableViewOperations.SelectFirstRow(listView);
                return;
            }

            listView.SelectRepObjectInfos(shelvesToSelect);

            if (listView.HasSelection())
                return;

            TableViewOperations.SelectDefaultRow(listView, defaultRow);

            if (listView.HasSelection())
                return;

            TableViewOperations.SelectFirstRow(listView);
        }

        internal static List<RepObjectInfo> GetSelectedRepObjectInfos(
            ShelvesListView listView)
        {
            return listView.GetSelectedRepObjectInfos();
        }

        internal static int GetSelectedShelvesCount(
            ShelvesListView listView)
        {
            return listView.GetSelection().Count;
        }

        internal static ChangesetInfo GetSelectedShelve(
            ShelvesListView listView)
        {
            List<RepObjectInfo> selectedRepObjectsInfos = listView.GetSelectedRepObjectInfos();

            if (selectedRepObjectsInfos.Count == 0)
                return null;

            return (ChangesetInfo)selectedRepObjectsInfos[0];
        }

        internal static List<ChangesetInfo> GetSelectedShelves(
            ShelvesListView listView)
        {
            return listView.GetSelectedRepObjectInfos().Cast<ChangesetInfo>().ToList();
        }

        internal static RepositorySpec GetSelectedRepository(
            ShelvesListView listView)
        {
            List<RepositorySpec> selectedRepositories = listView.GetSelectedRepositories();

            if (selectedRepositories.Count == 0)
                return null;

            return selectedRepositories[0];
        }

        internal static List<RepositorySpec> GetSelectedRepositories(
            ShelvesListView listView)
        {
            return listView.GetSelectedRepositories();
        }
    }
}
