using UnityEngine;
using Codice.CM.Common;
using PlasticGui.WorkspaceWindow;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI.StatusBar;
using PlasticGui.WorkspaceWindow.PendingChanges;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Developer
{
    internal class IncomingChangesNotification :
        StatusBar.IIncomingChangesNotification,
        CheckIncomingChanges.IUpdateIncomingChanges
    {
        internal IncomingChangesNotification(
            WorkspaceInfo wkInfo,
            IMergeViewLauncher mergeViewLauncher,
            PlasticWindow plasticWindow)
        {
            mWkInfo = wkInfo;
            mMergeViewLauncher = mergeViewLauncher;
            mPlasticWindow = plasticWindow;
        }

        internal void SetWorkspaceWindow(WorkspaceWindow workspaceWindow)
        {
            mWorkspaceWindow = workspaceWindow;
        }

        bool StatusBar.IIncomingChangesNotification.HasNotification
        {
            get { return mHasNotification; }
        }

        void StatusBar.IIncomingChangesNotification.OnGUI()
        {
            Texture2D icon = mData.Status == PlasticNotification.Status.Conflicts ?
                Images.GetConflictedIcon() :
                Images.GetOutOfSyncIcon();

            StatusBar.DrawIcon(icon);

            StatusBar.DrawNotification(new GUIContentNotification(
                new GUIContent(mData.InfoText)));

            if (StatusBar.DrawButton(new GUIContent(mData.ActionText, mData.TooltipText)))
            {
                if (mData.HasUpdateAction)
                {
                    mWorkspaceWindow.UpdateWorkspace();
                    return;
                }

                ShowIncomingChanges.FromNotificationBar(mWkInfo, mMergeViewLauncher);
            }
        }

        void CheckIncomingChanges.IUpdateIncomingChanges.Hide(WorkspaceInfo wkInfo)
        {
            if (!wkInfo.Equals(mWkInfo))
                return;

            PlasticPlugin.SetNotificationStatus(
                mPlasticWindow,
                PlasticNotification.Status.None);

            mData.Clear();

            mHasNotification = false;

            mPlasticWindow.Repaint();
        }

        void CheckIncomingChanges.IUpdateIncomingChanges.Show(
            WorkspaceInfo wkInfo,
            string infoText,
            string actionText,
            string tooltipText,
            CheckIncomingChanges.Severity severity,
            CheckIncomingChanges.Action action)
        {
            if (!wkInfo.Equals(mWkInfo))
                return;

            PlasticNotification.Status status = GetStatusFromSeverity(severity);

            mData.UpdateData(
                infoText,
                actionText,
                tooltipText,
                action == CheckIncomingChanges.Action.Update,
                status);

            mHasNotification = true;

            PlasticPlugin.SetNotificationStatus(
                mPlasticWindow,
                status);

            mPlasticWindow.Repaint();
        }

        static PlasticNotification.Status GetStatusFromSeverity(
            CheckIncomingChanges.Severity severity)
        {
            if (severity == CheckIncomingChanges.Severity.Info)
                return PlasticNotification.Status.IncomingChanges;

            if (severity == CheckIncomingChanges.Severity.Warning)
                return PlasticNotification.Status.Conflicts;

            return PlasticNotification.Status.None;
        }

        bool mHasNotification;
        StatusBar.IncomingChangesNotificationData mData =
            new StatusBar.IncomingChangesNotificationData();
        WorkspaceWindow mWorkspaceWindow;

        readonly WorkspaceInfo mWkInfo;
        readonly IMergeViewLauncher mMergeViewLauncher;
        readonly PlasticWindow mPlasticWindow;
    }
}
