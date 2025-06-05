using System;

using Codice.Client.Common;

using Unity.PlasticSCM.Editor.UI.Avatar;

using UnityEngine;

namespace Unity.PlasticSCM.Editor.UI
{
    internal static class DrawUserIcon
    {
        internal static void ForPendingChangesTab(
            ResolvedUser currentUser,
            Action avatarLoadedAction)
        {
            Rect rect = BuildUserIconAreaRect(35f);

            if (currentUser == null)
            {
                GUI.DrawTexture(rect, Images.GetEmptyGravatar());
                return;
            }

            GUI.Label(rect, new GUIContent(
                GetAvatar.ForEmail(currentUser.Name, avatarLoadedAction),
                currentUser.Name));
        }

        static Rect BuildUserIconAreaRect(float sizeOfImage)
        {
            GUIStyle commentTextAreaStyle = UnityStyles.PendingChangesTab.CommentTextArea;

            Rect result = GUILayoutUtility.GetRect(sizeOfImage, sizeOfImage); // Needs to be a square
            result.x = commentTextAreaStyle.margin.left;

            return result;
        }
    }
}
