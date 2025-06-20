use capitalize::Capitalize;
use corvutils::StringExtensions;
use singularity_client::{
	common::{
		ports::RecordResolver,
		presentation::DisplayName,
		stdb::{EventContext, ExternalActorReference, ExternalAuthenticationRequest},
	},
	domain::entities::{external_platform::SupportedExternalActorOrigin, message::MessageType},
};
use teloxide_core::types::{ChatId, InlineKeyboardButton, InlineKeyboardMarkup};

use super::OutboundTelegramActionRequest;
use crate::integrations::{
	commands::ExtAuthReqResolution,
	dtos::{ActionCommand, ActionKind},
	telegram::shared::constants::TELEGRAM_INLINE_BUTTON_CALLBACK_BYTE_LIMIT,
};

impl OutboundTelegramActionRequest {
	pub fn from_ext_auth_req(
		ctx: &EventContext, ext_auth_req: &ExternalAuthenticationRequest,
	) -> Result<Self, String> {
		let issuer_account = ext_auth_req
			.issuer
			.resolve(ctx)
			.ok_or("Unable to resolve issuer account.")?;

		let requester_account = ext_auth_req
			.requester
			.resolve(ctx)
			.ok_or("Unable to resolve requester account.")?;

		let ExternalActorReference {
			id: raw_user_id,
			origin,
		} = ext_auth_req
			.subject
			.parse()
			.map_err(|_| "Unable to parse subject account reference.")?;

		//* Double checking the platform tag
		//* In case of the forwarder letting it through unverified
		if origin.into_supported() != SupportedExternalActorOrigin::Telegram {
			return Err(format!("Origin {origin} does not match Telegram."));
		}

		let subject_user_id: ChatId = raw_user_id
			.parse()
			.map(|user_id: i64| ChatId(user_id))
			.map_err(|_| "Unable to parse subject user id.")?;

		let issuer_name = issuer_account.display_name(ctx);
		let requester_name = requester_account.display_name(ctx);

		// TODO: Abstract the choice mapping away, along with error handling
		let accept_choice = ExtAuthReqResolution::Accept(ext_auth_req.id);
		let reject_choice = ExtAuthReqResolution::Reject(ext_auth_req.id);

		let accept_callback_payload = ActionCommand {
			kind:    ActionKind::ExtAuthReqResolution,
			payload: accept_choice,
		}
		.try_to_string()?;

		let reject_callback_payload = ActionCommand {
			kind:    ActionKind::ExtAuthReqResolution,
			payload: reject_choice,
		}
		.try_to_string()?;

		if accept_callback_payload.len() > TELEGRAM_INLINE_BUTTON_CALLBACK_BYTE_LIMIT {
			return Err(format!(
				r#"
					Telegram callback payload cannot exceed
					{TELEGRAM_INLINE_BUTTON_CALLBACK_BYTE_LIMIT} bytes,
					but the current length of `accept_callback_payload` is {length} bytes
				"#,
				length = accept_callback_payload.len()
			)
			.squash_whitespace());
		}

		if reject_callback_payload.len() > TELEGRAM_INLINE_BUTTON_CALLBACK_BYTE_LIMIT {
			return Err(format!(
				r#"
					Telegram callback payload cannot exceed
					{TELEGRAM_INLINE_BUTTON_CALLBACK_BYTE_LIMIT} bytes,
					but the current length of `reject_callback_payload` is {length} bytes.
				"#,
				length = reject_callback_payload.len()
			)
			.squash_whitespace());
		}

		Ok(OutboundTelegramActionRequest {
			chat_id:             subject_user_id,
			thread_id:           None,
			reply_to_message_id: None,

			text: format!(
				"{}\n\n{}",
				format!(
					"{} <strong>{issuer_name}</strong>",
					MessageType::symbol_by_account_role(Some(issuer_account.role)),
				),
				format!(
					r#"
						{requester_name} has requested to link this {platform_name} account.
						If you are the not {requester_name}, please reject this request.
					"#,
					platform_name = SupportedExternalActorOrigin::Telegram
						.to_string()
						.capitalize()
				)
				.squash_whitespace(),
			),

			reply_markup: InlineKeyboardMarkup::new([[
				InlineKeyboardButton::callback(accept_choice.label(), accept_callback_payload),
				InlineKeyboardButton::callback(reject_choice.label(), reject_callback_payload),
			]]),
		})
	}
}
