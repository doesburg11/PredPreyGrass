using UnityEditor;
using UnityEngine;

using PlasticGui;
using PlasticGui.WorkspaceWindow.QueryViews.Shelves;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Views.Shelves
{
    internal class ShelvesViewMenu
    {
        internal GenericMenu Menu { get { return mMenu; } }

        internal ShelvesViewMenu(IShelveMenuOperations shelveMenuOperations)
        {
            mShelveMenuOperations = shelveMenuOperations;

            BuildComponents();
        }

        internal void Popup()
        {
            mMenu = new GenericMenu();

            UpdateMenuItems(mMenu);

            mMenu.ShowAsContext();
        }

        internal bool ProcessKeyActionIfNeeded(Event e)
        {
            ShelveMenuOperations operationToExecute = GetMenuOperations(e);

            if (operationToExecute == ShelveMenuOperations.None)
                return false;

            ShelveMenuOperations operations = ShelveMenuUpdater.GetAvailableMenuOperations(
                mShelveMenuOperations.GetSelectedShelvesCount());

            if (!operations.HasFlag(operationToExecute))
                return false;

            ProcessMenuOperation(operationToExecute);
            return true;
        }

        void ApplyShelveInWorkspace_Click()
        {
            mShelveMenuOperations.ApplyShelveInWorkspace();
        }

        void DeleteShelve_Click()
        {
            mShelveMenuOperations.DeleteShelve();
        }

        void OpenShelveInNewWindow_Click()
        {
            mShelveMenuOperations.OpenSelectedShelveInNewWindow();
        }

        void UpdateMenuItems(GenericMenu menu)
        {
            ShelveMenuOperations operations = ShelveMenuUpdater.GetAvailableMenuOperations(
                mShelveMenuOperations.GetSelectedShelvesCount());

            AddShelveMenuItem(
                mApplyShelveInWorkspaceMenuItemContent,
                menu,
                operations,
                ShelveMenuOperations.ApplyShelveInWorkspace,
                ApplyShelveInWorkspace_Click);

            AddShelveMenuItem(
                mDeleteShelveMenuItemContent,
                menu,
                operations,
                ShelveMenuOperations.Delete,
                DeleteShelve_Click);

            menu.AddSeparator(string.Empty);

            AddShelveMenuItem(
                mOpenShelveInNewWindowMenuItemContent,
                menu,
                operations,
                ShelveMenuOperations.ViewShelve,
                OpenShelveInNewWindow_Click);
        }

        void ProcessMenuOperation(
            ShelveMenuOperations operationToExecute)
        {
            if (operationToExecute == ShelveMenuOperations.ApplyShelveInWorkspace)
            {
                ApplyShelveInWorkspace_Click();
                return;
            }

            if (operationToExecute == ShelveMenuOperations.Delete)
            {
                DeleteShelve_Click();
                return;
            }

            if (operationToExecute == ShelveMenuOperations.ViewShelve)
            {
                OpenShelveInNewWindow_Click();
                return;
            }
        }

        static void AddShelveMenuItem(
            GUIContent menuItemContent,
            GenericMenu menu,
            ShelveMenuOperations operations,
            ShelveMenuOperations operationsToCheck,
            GenericMenu.MenuFunction menuFunction)
        {
            if (operations.HasFlag(operationsToCheck))
            {
                menu.AddItem(
                    menuItemContent,
                    false,
                    menuFunction);

                return;
            }

            menu.AddDisabledItem(menuItemContent);
        }

        static ShelveMenuOperations GetMenuOperations( Event e)
        {
            if (Keyboard.IsKeyPressed(e, KeyCode.Delete))
                return ShelveMenuOperations.Delete;

            return ShelveMenuOperations.None;
        }

        void BuildComponents()
        {
            mApplyShelveInWorkspaceMenuItemContent = new GUIContent(
                PlasticLocalization.Name.ShelveMenuItemApplyShelveInWorkspace.GetString());

            mDeleteShelveMenuItemContent = new GUIContent(string.Format("{0} {1}",
                PlasticLocalization.Name.ShelveMenuItemDeleteShelve.GetString(),
                GetPlasticShortcut.ForDelete()));

            mOpenShelveInNewWindowMenuItemContent = new GUIContent(
                PlasticLocalization.Name.ShelveMenuItemOpenShelveInNewWindow.GetString());
        }

        GenericMenu mMenu;

        GUIContent mApplyShelveInWorkspaceMenuItemContent;
        GUIContent mDeleteShelveMenuItemContent;
        GUIContent mOpenShelveInNewWindowMenuItemContent;

        readonly IShelveMenuOperations mShelveMenuOperations;
    }
}
