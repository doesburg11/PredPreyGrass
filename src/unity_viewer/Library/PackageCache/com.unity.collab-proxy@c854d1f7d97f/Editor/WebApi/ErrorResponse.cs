using System.ComponentModel;
using Unity.Plastic.Newtonsoft.Json;

// Warning: keep in sync with src/client/clientcommon/webapi/responses/ErrorResponse.cs
// This copy preserves the old namespace that is part of the public API of the package.
namespace PlasticGui.WebApi.Responses
{
    // Internal usage. This isn't a public API.
    [EditorBrowsable(EditorBrowsableState.Never)]
    public class ErrorResponse
    {
        // Internal usage. This isn't a public API.
        [EditorBrowsable(EditorBrowsableState.Never)]
        [JsonProperty("error")]
        public ErrorFields Error { get; set; }

        // Internal usage. This isn't a public API.
        [EditorBrowsable(EditorBrowsableState.Never)]
        public class ErrorFields
        {
            // Internal usage. This isn't a public API.
            [EditorBrowsable(EditorBrowsableState.Never)]
            public string ErrorCode { get; set; }

            // Internal usage. This isn't a public API.
            [EditorBrowsable(EditorBrowsableState.Never)]
            public string Message { get; set; }
        }
    }
}
