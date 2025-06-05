using UnityEditor;
using UnityEngine;

using Codice.CM.Common;
using PlasticGui;
using PlasticGui.WorkspaceWindow;
using PlasticGui.WorkspaceWindow.Topbar;
using PlasticGui.WorkspaceWindow.Merge;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.StatusBar;

namespace Unity.PlasticSCM.Editor.Developer
{
    internal class ShelvedChangesNotification :
        StatusBar.IShelvedChangesNotification,
        CheckShelvedChanges.IUpdateShelvedChangesNotification
    {
        internal ShelvedChangesNotification(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            ViewSwitcher viewSwitcher,
            PlasticWindow plasticWindow)
        {
            mWkInfo = wkInfo;
            mRepSpec = repSpec;
            mViewSwitcher = viewSwitcher;
            mPlasticWindow = plasticWindow;
        }

        bool StatusBar.IShelvedChangesNotification.HasNotification
        {
            get { return mHasNotification; }
        }

        void StatusBar.IShelvedChangesNotification.SetWorkspaceWindow(
            WorkspaceWindow workspaceWindow)
        {
            mWorkspaceWindow = workspaceWindow;
        }

        void StatusBar.IShelvedChangesNotification.SetShelvedChangesUpdater(
            IShelvedChangesUpdater shelvedChangesUpdater)
        {
            mShelvedChangesUpdater = shelvedChangesUpdater;
        }

        void StatusBar.IShelvedChangesNotification.OnGUI()
        {
            Texture2D icon = Images.GetInfoBellNotificationIcon();

            StatusBar.DrawIcon(icon, UnityConstants.STATUS_BAR_ICON_SIZE - 2);

            StatusBar.DrawNotification(
                new GUIContentNotification(new GUIContent(
                    PlasticLocalization.Name.ShelvedChanges.GetString(),
                    PlasticLocalization.Name.ShelvedChangesExplanation.GetString())));

            GenericMenu discardShelveDropdownMenu = new GenericMenu();
            discardShelveDropdownMenu.AddItem(
                new GUIContent(PlasticLocalization.Name.DiscardShelvedChanges.GetString()),
                false,
                () =>
                {
                    ShelvedChangesNotificationPanelOperations.DiscardShelvedChanges(
                        mWkInfo,
                        mShelveInfo,
                        this,
                        mShelvedChangesUpdater,
                        mViewSwitcher,
                        mWorkspaceWindow);
                });

            DrawActionButtonWithMenu.For(
                PlasticLocalization.Name.ViewButton.GetString(),
                PlasticLocalization.Name.ViewShelvedChangesButtonExplanation.GetString(),
                () =>
                {
                    if (mShelveInfo == null || mViewSwitcher == null)
                        return;

                    ((IMergeViewLauncher)mViewSwitcher).MergeFrom(
                        mRepSpec,
                        mShelveInfo,
                        EnumMergeType.ChangesetCherryPick,
                        showDiscardChangesButton: true);
                },
                discardShelveDropdownMenu);
        }

        void CheckShelvedChanges.IUpdateShelvedChangesNotification.Hide(
            WorkspaceInfo wkInfo)
        {
            if (!wkInfo.Equals(mWkInfo))
                return;

            mShelveInfo = null;

            mHasNotification = false;

            mPlasticWindow.Repaint();
        }

        void CheckShelvedChanges.IUpdateShelvedChangesNotification.Show(
            WorkspaceInfo wkInfo,
            RepositorySpec repSpec,
            ChangesetInfo shelveInfo)
        {
            if (!wkInfo.Equals(mWkInfo))
                return;

            mShelveInfo = shelveInfo;

            mHasNotification = true;

            mPlasticWindow.Repaint();
        }

        bool mHasNotification;
        ChangesetInfo mShelveInfo;

        WorkspaceWindow mWorkspaceWindow;
        IShelvedChangesUpdater mShelvedChangesUpdater;

        readonly WorkspaceInfo mWkInfo;
        readonly RepositorySpec mRepSpec;
        readonly ViewSwitcher mViewSwitcher;
        readonly PlasticWindow mPlasticWindow;
    }
}
