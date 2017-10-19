# Generated by stubbs:add-option. Do not edit, if using stubbs.
# Created: Tue Jan 29 12:05:48 PST 2013
#
#/ usage: chef-solo:register-server  --key-repo-url <>  --key-repo-id <>  --key-repo-username <>  --key-repo-password <> 

# _rerun_options_parse_ - Parse the command arguments and set option variables.
#
#     rerun_options_parse "$@"
#
# Arguments:
#
# * the command options and their arguments
#
# Notes:
# 
# * Sets shell variables for any parsed options.
# * The "-?" help argument prints command usage and will exit 2.
# * Return 0 for successful option parse.
#
rerun_options_parse() {

    while [ "$#" -gt 0 ]; do
        OPT="$1"
        case "$OPT" in
            --key-repo-url) rerun_option_check $# $1; KEY_REPO_URL=$2 ; shift ;;
            --key-repo-id) rerun_option_check $# $1; KEY_REPO_ID=$2 ; shift ;;
            --key-repo-username) rerun_option_check $# $1; KEY_REPO_USERNAME=$2 ; shift ;;
            --key-repo-password) rerun_option_check $# $1; KEY_REPO_PASSWORD=$2 ; shift ;;
            # help option
            -?)
                rerun_option_usage
                exit 2
                ;;
            # end of options, just arguments left
            *)
              break
        esac
        shift
    done

    # Set defaultable options.

    # Check required options are set
    [ -z "$KEY_REPO_URL" ] && { echo >&2 "missing required option: --key-repo-url" ; return 2 ; }
    [ -z "$KEY_REPO_ID" ] && { echo >&2 "missing required option: --key-repo-id" ; return 2 ; }
    [ -z "$KEY_REPO_USERNAME" ] && { echo >&2 "missing required option: --key-repo-username" ; return 2 ; }
    [ -z "$KEY_REPO_PASSWORD" ] && { echo >&2 "missing required option: --key-repo-password" ; return 2 ; }
    # If option variables are declared exportable, export them.

    #
    return 0
}


# Initialize the options variables to null.
KEY_REPO_URL=
KEY_REPO_ID=
KEY_REPO_USERNAME=
KEY_REPO_PASSWORD=


