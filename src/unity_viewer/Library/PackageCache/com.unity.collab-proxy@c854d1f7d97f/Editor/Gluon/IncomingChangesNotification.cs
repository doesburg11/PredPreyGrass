using UnityEngine;
using Codice.CM.Common;
using PlasticGui.Gluon.WorkspaceWindow;
using PlasticGui.Gluon;
using Unity.PlasticSCM.Editor.UI.StatusBar;
using Unity.PlasticSCM.Editor.UI;

namespace Unity.PlasticSCM.Editor.Gluon
{
    internal class IncomingChangesNotification :
        StatusBar.IIncomingChangesNotification,
        CheckIncomingChanges.IUpdateIncomingChanges
    {
        internal IncomingChangesNotification(
            WorkspaceInfo wkInfo,
            IGluonViewSwitcher gluonViewSwitcher,
            PlasticWindow plasticWindow)
        {
            mWkInfo = wkInfo;
            mGluonViewSwitcher = gluonViewSwitcher;
            mPlasticWindow = plasticWindow;
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
                ShowIncomingChanges.FromNotificationBar(mWkInfo, mGluonViewSwitcher);
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
            CheckIncomingChanges.Severity severity)
        {
            if (!wkInfo.Equals(mWkInfo))
                return;

            PlasticNotification.Status status = GetStatusFromSeverity(severity);

            mData.UpdateData(
                infoText,
                actionText,
                tooltipText,
                false,
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

        readonly WorkspaceInfo mWkInfo;
        readonly IGluonViewSwitcher mGluonViewSwitcher;
        readonly PlasticWindow mPlasticWindow;
    }
}
