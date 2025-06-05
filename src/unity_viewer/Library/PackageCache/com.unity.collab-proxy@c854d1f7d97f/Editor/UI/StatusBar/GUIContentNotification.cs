using UnityEngine;

namespace Unity.PlasticSCM.Editor.UI.StatusBar
{
    internal class GUIContentNotification : INotificationContent
    {
        internal GUIContentNotification(string content) : this(new GUIContent(content)) { }

        internal GUIContentNotification(GUIContent content)
        {
            mGUIContent = content;
        }

        void INotificationContent.OnGUI()
        {
            GUILayout.Label(
                mGUIContent,
                UnityStyles.StatusBar.NotificationLabel);
        }

        readonly GUIContent mGUIContent;
    }
}
