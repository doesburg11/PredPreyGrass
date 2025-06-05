using System;
using System.Collections.Generic;

using UnityEditor;
using UnityEngine;

using Codice.Client.Common;
using Codice.CM.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.Tree;

namespace Unity.PlasticSCM.Editor.Views.PendingChanges
{
    internal static class DrawPendingChangesEmptyState
    {
        internal static void ForNotifySuccessData(
            Rect rect,
            CreatedChangesetData data,
            Action openLink,
            Action copyLink)
        {
            DrawTreeViewEmptyState.DrawCenteredOnRect(rect, () =>
            {
                DrawNotifySuccessData(data, openLink, copyLink);
            });
        }

        internal static void ForNotifySuccessDataWithInviteMembers(
            Rect rect,
            CreatedChangesetData data,
            Action openLink,
            Action copyLink)
        {
            GUIContent linkContent = new GUIContent(
                PlasticLocalization.Name.InviteOtherTeamMembers.GetString());

            DrawTreeViewEmptyState.DrawCenteredOnRect(rect, () =>
            {
                GUILayout.BeginVertical();

                GUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                DrawNotifySuccessData(
                    data,
                    openLink,
                    copyLink);
                GUILayout.FlexibleSpace();
                GUILayout.EndHorizontal();

                GUILayout.Space(UnityConstants.TREEVIEW_STATUS_CONTENT_PADDING);

                GUILayout.BeginHorizontal();
                GUILayout.FlexibleSpace();
                DrawInviteMembersLink(data.RepositorySpec, linkContent);
                GUILayout.FlexibleSpace();
                GUILayout.EndHorizontal();

                GUILayout.EndVertical();
            });
        }

        static void DrawInviteMembersLink(
            RepositorySpec repSpec,
            GUIContent linkContent)
        {
            if (GUILayout.Button(linkContent, EditorStyles.linkLabel))
            {
                OpenInviteUsersPage.Run(repSpec, UnityUrl.UnityDashboard.UnityCloudRequestSource.Editor);
            }

            EditorGUIUtility.AddCursorRect(GUILayoutUtility.GetLastRect(), MouseCursor.Link);
        }

        static void DrawNotifySuccessData(
            CreatedChangesetData data,
            Action openLink,
            Action copyLink)
        {
            if (data.OperationType == CreatedChangesetData.Type.Checkin)
            {
                DrawCheckinSuccessMessage(
                    data.CreatedChangesetId,
                    openLink,
                    copyLink);
                return;
            }

            DrawShelveSuccessMessage(
                data.CreatedChangesetId,
                openLink,
                copyLink);
        }

        static void DrawCheckinSuccessMessage(
            long changesetId,
            Action openChangesetLink,
            Action copyChangesetLink)
        {
            string text = string.Concat(
                PlasticLocalization.Name.CheckinCompleted.GetString(),
                " ",
                "{0} " + PlasticLocalization.Name.CheckinChangesetWasCreatedPart.GetString());

            string linkText =
                string.Format("{0} {1}",
                PlasticLocalization.Name.Changeset.GetString(),
                changesetId.ToString());

            DrawCreatedChangesetMessage(
                text,
                linkText,
                openChangesetLink,
                copyChangesetLink);
        }

        static void DrawShelveSuccessMessage(
            long shelvesetId,
            Action openShelveLink,
            Action copyShelveLink)
        {
            string text = PlasticLocalization.Name.ShelveCreatedMessage.GetString() + ".";
            string linkText = string.Format("{0} {1}",
                PlasticLocalization.Name.Shelve.GetString().ToLower(),
                Math.Abs(shelvesetId).ToString());

            DrawCreatedChangesetMessage(
                text,
                linkText,
                openShelveLink,
                copyShelveLink);
        }

        static void DrawCreatedChangesetMessage(
            string text,
            string actionText,
            Action openLink,
            Action copyLink)
        {
            GUILayout.Label(
                Images.GetStepOkIcon(),
                UnityStyles.Tree.StatusLabel,
                GUILayout.Width(UnityConstants.TREEVIEW_STATUS_ICON_SIZE),
                GUILayout.Height(UnityConstants.TREEVIEW_STATUS_ICON_SIZE));

            GUILayout.Space(UnityConstants.TREEVIEW_STATUS_CONTENT_PADDING);

            DrawTextBlockWithLink.ForMultiLinkLabel(
                new MultiLinkLabelData(
                    text,
                    new List<string> { actionText },
                    new List<Action> { openLink }),
                UnityStyles.Tree.StatusLabel);

            GUILayout.Space(UnityConstants.TREEVIEW_STATUS_CONTENT_PADDING);

            if (GUILayout.Button(
                new GUIContent(
                    Images.GetClipboardIcon(),
                    PlasticLocalization.Name.DiffLinkButtonTooltip.GetString()),
                UnityStyles.Tree.CopyToClipboardButton))
            {
                copyLink();
            }
        }
    }
}
