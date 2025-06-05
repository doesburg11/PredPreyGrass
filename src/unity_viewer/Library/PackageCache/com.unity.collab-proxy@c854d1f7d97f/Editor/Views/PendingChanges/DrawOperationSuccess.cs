using System;

using UnityEditor;
using UnityEngine;

using Codice.Client.Common.Threading;
using Codice.CM.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.Tool;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.StatusBar;
using Unity.PlasticSCM.Editor.UI.Tree;
using Unity.PlasticSCM.Editor.WebApi;

namespace Unity.PlasticSCM.Editor.Views.PendingChanges
{
    internal interface IDrawOperationSuccess
    {
        void InStatusBar(StatusBar statusBar);
        void InEmptyState(Rect rect);
    }

    internal class NotifySuccessForCreatedChangeset : IDrawOperationSuccess
    {
        internal NotifySuccessForCreatedChangeset(
            CreatedChangesetData createdChangesetData,
            Action openLink,
            Action copyLink,
            Action repaint)
        {
            mCreatedChangesetData = createdChangesetData;
            mOpenLink = openLink;
            mCopyLink = copyLink;
            mRepaint = repaint;
        }

        void IDrawOperationSuccess.InStatusBar(StatusBar statusBar)
        {
            INotificationContent notificationContent = new PendingChangesStatusSuccessNotificationContent(
                mCreatedChangesetData,
                mOpenLink,
                mCopyLink);

            statusBar.Notify(
                notificationContent,
                MessageType.None,
                Images.GetStepOkIcon());
        }

        void IDrawOperationSuccess.InEmptyState(Rect rect)
        {
            if (!mCanInviteMembersFromPendingChangesAlreadyCalculated &&
                mCreatedChangesetData.OperationType == CreatedChangesetData.Type.Checkin)
            {
                EnableInviteMembersIfFirstCheckinAndAdmin(mCreatedChangesetData.RepositorySpec.Server);
                mCanInviteMembersFromPendingChangesAlreadyCalculated = true;
            }

            mEmptyStateLastValidRect = EmptyStateData.EnsureValidRect(
                rect, mEmptyStateLastValidRect, Event.current.type, mRepaint);

            if (mCanInviteMembersFromPendingChanges)
            {
                DrawPendingChangesEmptyState.ForNotifySuccessDataWithInviteMembers(
                    mEmptyStateLastValidRect,
                    mCreatedChangesetData,
                    mOpenLink,
                    mCopyLink);
                return;
            }

            DrawPendingChangesEmptyState.ForNotifySuccessData(
                mEmptyStateLastValidRect,
                mCreatedChangesetData,
                mOpenLink,
                mCopyLink);
        }

        void EnableInviteMembersIfFirstCheckinAndAdmin(string server)
        {
            if (!PlasticGui.Plastic.API.IsCloud(server))
                return;

            bool isFirstCheckin = !BoolSetting.Load(
                UnityConstants.FIRST_CHECKIN_SUBMITTED, false);

            if (!isFirstCheckin)
                return;

            BoolSetting.Save(true, UnityConstants.FIRST_CHECKIN_SUBMITTED);

            string organizationName = ServerOrganizationParser.GetOrganizationFromServer(server);

            CurrentUserAdminCheckResponse response = null;

            IThreadWaiter waiter = ThreadWaiter.GetWaiter(50);
            waiter.Execute(
                /*threadOperationDelegate*/
                delegate
                {
                    string authToken = AuthToken.GetForServer(server);

                    if (string.IsNullOrEmpty(authToken))
                        return;

                    response = WebRestApiClient.PlasticScm.IsUserAdmin(organizationName, authToken);
                },
                /*afterOperationDelegate*/
                delegate
                {
                    if (response == null || !response.IsCurrentUserAdmin)
                        return;

                    mCanInviteMembersFromPendingChanges = true;

                    mRepaint();
                });
        }

        bool mCanInviteMembersFromPendingChangesAlreadyCalculated;
        bool mCanInviteMembersFromPendingChanges;
        Rect mEmptyStateLastValidRect;

        readonly Action mRepaint;
        readonly Action mCopyLink;
        readonly Action mOpenLink;
        readonly CreatedChangesetData mCreatedChangesetData;
    }

    internal class NotifySuccessForUndo : IDrawOperationSuccess
    {
        internal NotifySuccessForUndo(Action repaint)
        {
            mRepaint = repaint;
        }

        void IDrawOperationSuccess.InStatusBar(StatusBar statusBar)
        {
            INotificationContent notificationContent = new GUIContentNotification(
                PlasticLocalization.Name.UndoCompleted.GetString());

            statusBar.Notify(
                notificationContent,
                MessageType.None,
                Images.GetStepOkIcon());
        }

        void IDrawOperationSuccess.InEmptyState(Rect rect)
        {
            mEmptyStateData.Update(
                PlasticLocalization.Name.UndoCompleted.GetString(),
                rect, Event.current.type, mRepaint);

            DrawTreeViewEmptyState.For(Images.GetStepOkIcon(), mEmptyStateData);
        }

        readonly EmptyStateData mEmptyStateData = new EmptyStateData();
        readonly Action mRepaint;
    }
}
