using System;
using System.Collections.Generic;

using Codice.Client.Common;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI;
using Unity.PlasticSCM.Editor.UI.StatusBar;

using UnityEngine;

namespace Unity.PlasticSCM.Editor.Views.PendingChanges
{
    internal class PendingChangesStatusSuccessNotificationContent : INotificationContent
    {
        internal PendingChangesStatusSuccessNotificationContent(
            CreatedChangesetData data,
            Action openLink,
            Action copyLink)
        {
            mCreatedChangesetData = data;
            mOpenLinkAction = openLink;
            mCopyAction = copyLink;
        }

        void INotificationContent.OnGUI()
        {
            if (mCreatedChangesetData.OperationType == CreatedChangesetData.Type.Checkin)
            {
                DrawCheckinSuccessMessage(
                    mCreatedChangesetData.CreatedChangesetId,
                    mOpenLinkAction,
                    mCopyAction);
                return;
            }

            DrawShelveSuccessMessage(
                mCreatedChangesetData.CreatedChangesetId,
                mOpenLinkAction,
                mCopyAction);
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
            string linkText,
            Action openLink,
            Action copyLink)
        {
            GUILayout.BeginHorizontal();

            GUILayout.Space(2);

            DrawTextBlockWithLink.ForMultiLinkLabel(
                new MultiLinkLabelData(
                    text,
                    new List<string> { linkText },
                    new List<Action> { openLink }),
                UnityStyles.StatusBar.NotificationLabel);

            GUILayout.Space(4);

            if (GUILayout.Button(
                new GUIContent(
                    Images.GetClipboardIcon(),
                    PlasticLocalization.Name.DiffLinkButtonTooltip.GetString()),
                UnityStyles.StatusBar.CopyToClipboardButton))
            {
                copyLink();
            }

            GUILayout.EndHorizontal();
        }

        readonly CreatedChangesetData mCreatedChangesetData;
        readonly Action mOpenLinkAction;
        readonly Action mCopyAction;
    }
}
