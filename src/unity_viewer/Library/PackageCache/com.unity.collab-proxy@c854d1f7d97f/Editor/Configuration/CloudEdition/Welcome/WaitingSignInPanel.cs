using System;

using Codice.Client.Common;
using Codice.Client.Common.Authentication;
using Codice.Client.Common.WebApi;
using Codice.CM.Common;

using PlasticGui;
using Unity.PlasticSCM.Editor.UI.UIElements;
using UnityEngine.UIElements;

namespace Unity.PlasticSCM.Editor.Configuration.CloudEdition.Welcome
{
    internal class WaitingSignInPanel : VisualElement
    {
        internal WaitingSignInPanel(
            IWelcomeWindowNotify parentNotify,
            OAuthSignIn.INotify notify,
            IPlasticWebRestApi restApi)
        {
            mParentNotify = parentNotify;

            mNotify = notify;
            mRestApi = restApi;

            InitializeLayoutAndStyles();

            BuildComponents();
        }

        internal void OAuthSignIn(
            AuthProvider provider,
            IGetCredentialsFromState getCredentialsFromState)
        {
            mSignIn = new OAuthSignIn();

            mSignIn.SignInForProviderInThreadWaiter(
                provider,
                string.Empty,
                mProgressControls,
                mNotify,
                new OAuthSignIn.Browser(),
                getCredentialsFromState);

            ShowWaitingSpinner();
        }

        internal void OnAutoLogin()
        {
            mCompleteOnBrowserLabel.visible = false;
            mCancelButton.visible = false;

            mProgressControls.ProgressData.ProgressMessage =
                PlasticLocalization.Name.SigningIn.GetString();

            ShowWaitingSpinner();
        }

        internal void Dispose()
        {
            mCancelButton.clicked -= CancelButton_Clicked;
        }

        void InitializeLayoutAndStyles()
        {
            this.LoadLayout(typeof(WaitingSignInPanel).Name);
            this.LoadStyle(typeof(WaitingSignInPanel).Name);
        }

        void ShowWaitingSpinner()
        {
            var spinner = new LoadingSpinner();
            mProgressContainer.Add(spinner);
            spinner.Start();

            var checkinMessageLabel = new Label(mProgressControls.ProgressData.ProgressMessage);
            checkinMessageLabel.style.paddingLeft = 20;
            mProgressContainer.Add(checkinMessageLabel);
        }

        void CancelButton_Clicked()
        {
            mSignIn.Cancel();
            mParentNotify.Back();
        }

        void BuildComponents()
        {
            this.SetControlText<Label>("signInToPlasticSCM",
                PlasticLocalization.Name.SignInToUnityVCS);

            mCompleteOnBrowserLabel = this.Q<Label>("completeSignInOnBrowser");
            mCompleteOnBrowserLabel.text = PlasticLocalization.Name.CompleteSignInOnBrowser.GetString();

            mProgressContainer = this.Q<VisualElement>("progressContainer");

            mProgressControls = new UI.Progress.ProgressControlsForDialogs();

            mCancelButton = this.Query<Button>("cancelButton");
            mCancelButton.text = PlasticLocalization.GetString(
                PlasticLocalization.Name.CancelButton);
            mCancelButton.visible = true;
            mCancelButton.clicked += CancelButton_Clicked;
        }

        Button mCancelButton;
        VisualElement mProgressContainer;
        Label mCompleteOnBrowserLabel;

        OAuthSignIn mSignIn;

        UI.Progress.ProgressControlsForDialogs mProgressControls;

        readonly IPlasticWebRestApi mRestApi;
        readonly OAuthSignIn.INotify mNotify;
        readonly IWelcomeWindowNotify mParentNotify;
    }
}
