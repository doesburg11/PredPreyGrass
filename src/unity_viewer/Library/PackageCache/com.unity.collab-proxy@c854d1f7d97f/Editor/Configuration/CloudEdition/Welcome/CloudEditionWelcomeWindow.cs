using System.Collections.Generic;

using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

using Codice.Client.Common;
using Codice.Client.Common.Authentication;
using Codice.Client.Common.Connection;
using Codice.Client.Common.WebApi;
using Codice.Client.Common.WebApi.Responses;
using Codice.CM.Common;
using Codice.LogWrapper;
using PlasticGui;
using Unity.PlasticSCM.Editor.UI.Progress;
using Unity.PlasticSCM.Editor.Views.Welcome;

namespace Unity.PlasticSCM.Editor.Configuration.CloudEdition.Welcome
{
    internal interface IWelcomeWindowNotify
    {
        void ProcessLoginResponseWithOrganizations(Credentials credentials, List<string> cloudServers);

        void Back();
    }

    internal class CloudEditionWelcomeWindow :
        EditorWindow, IWelcomeWindowNotify, OAuthSignIn.INotify, GetCloudOrganizations.INotify
    {
        internal static void ShowWindow(
            IPlasticWebRestApi restApi,
            WelcomeView welcomeView,
            bool autoLogin = false)
        {
            sRestApi = restApi;
            sAutoLogin = autoLogin;
            CloudEditionWelcomeWindow window = GetWindow<CloudEditionWelcomeWindow>();

            window.titleContent = new GUIContent(
                PlasticLocalization.GetString(PlasticLocalization.Name.SignInToUnityVCS));
            window.minSize = window.maxSize = new Vector2(450, 300);

            window.mWelcomeView = welcomeView;

            window.Show();
        }

        internal static CloudEditionWelcomeWindow GetWelcomeWindow()
        {
            return GetWindow<CloudEditionWelcomeWindow>();
        }

        internal void CancelJoinOrganization()
        {
            if (sAutoLogin)
            {
                mLog.Debug("CancelJoinOrganization");
                GetWindow<PlasticWindow>().GetWelcomeView().autoLoginState = AutoLogin.State.Started;
            }
        }

        internal void ReplaceRootPanel(VisualElement panel)
        {
            rootVisualElement.Clear();
            rootVisualElement.Add(panel);
        }

        internal SignInPanel GetSignInPanel()
        {
            return mSignInPanel;
        }

        internal void GetOrganizations(Credentials credentials)
        {
            mCredentials = credentials;

            GetCloudOrganizations.GetOrganizationsInThreadWaiter(
                mCredentials.User.Data,
                mCredentials.User.Password,
                new ProgressControlsForDialogs(),
                this,
                sRestApi,
                CmConnection.Get());
        }

        void ShowOrganizationsPanel(
            Credentials credentials, List<string> cloudServers, string errorMessage)
        {
            mLog.DebugFormat("ShowOrganizationsPanel({0}, {1} orgs) {2}",
                credentials.Mode, cloudServers.Count, errorMessage);

            mOrganizationPanel = new OrganizationPanel(
                this,
                cloudServers,
                errorMessage,
                GetWindowTitle(),
                joinedOrganization =>
                {
                    mLog.DebugFormat("JoinedOrganization: {0}", joinedOrganization);
                    ClientConfiguration.Save(
                        joinedOrganization,
                        credentials.Mode,
                        credentials.User.Data,
                        credentials.User.Password);
                });

            ReplaceRootPanel(mOrganizationPanel);
        }

        string GetWindowTitle()
        {
            return PlasticLocalization.Name.SignInToUnityVCS.GetString();
        }

        void OnEnable()
        {
            BuildComponents();
        }

        void OnDestroy()
        {
            Dispose();

            if (mWelcomeView != null)
                mWelcomeView.OnUserClosedConfigurationWindow();
        }

        void Dispose()
        {
            if (mSignInPanel != null)
                mSignInPanel.Dispose();

            if (mOrganizationPanel != null)
                mOrganizationPanel.Dispose();
        }

        // Used by WaitingSignInPanel
        void OAuthSignIn.INotify.SignedInForCloud(
            string chosenProviderName, Credentials credentials)
        {
            mLog.Debug("SignedInForCloud");

            GetOrganizations(credentials);
        }

        void OAuthSignIn.INotify.SignedInForOnPremise(
            string server, string proxy, Credentials credentials)
        {
            // Won't run
        }

        void OAuthSignIn.INotify.Cancel(string message)
        {
            mLog.Debug("Cancel");
            Focus();
        }

        void GetCloudOrganizations.INotify.CloudOrganizationsRetrieved(List<string> cloudServers)
        {
            ShowOrganizationsPanel(mCredentials, cloudServers, errorMessage: null);
        }

        void GetCloudOrganizations.INotify.Error(ErrorResponse.ErrorFields error)
        {
            ShowOrganizationsPanel(mCredentials, cloudServers: null, errorMessage: error.Message);
        }

        void IWelcomeWindowNotify.ProcessLoginResponseWithOrganizations(
            Credentials credentials, List<string> cloudServers)
        {
            mLog.DebugFormat("ProcessLoginResponseWithOrganizations: {0} orgs", cloudServers.Count);
            mCredentials = credentials;
            ShowOrganizationsPanel(mCredentials, cloudServers, errorMessage: null);
        }

        void IWelcomeWindowNotify.Back()
        {
            rootVisualElement.Clear();
            rootVisualElement.Add(mSignInPanel);
        }

        void BuildComponents()
        {
            VisualElement root = rootVisualElement;

            root.Clear();

            mSignInPanel = new SignInPanel(this, sRestApi);

            titleContent = new GUIContent(GetWindowTitle());

            root.Add(mSignInPanel);

            if (sAutoLogin)
            {
                mSignInPanel.SignInWithUnityIdButtonAutoLogin();
            }
        }

        string mUserName;
        Credentials mCredentials;

        OrganizationPanel mOrganizationPanel;
        SignInPanel mSignInPanel;
        WelcomeView mWelcomeView;

        static IPlasticWebRestApi sRestApi;
        static bool sAutoLogin = false;

        static readonly ILog mLog = PlasticApp.GetLogger("CloudEditionWelcomeWindow");
    }
}
