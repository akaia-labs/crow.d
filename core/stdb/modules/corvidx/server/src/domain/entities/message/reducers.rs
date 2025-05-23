use spacetimedb::{ReducerContext, Table, reducer};

use super::{model::*, validation::*};
use crate::{common::ports::RecordResolution, domain::entities::tp_account::TpAccountReference};

#[reducer]
/// Facilitates the basic internal messaging functionality
pub fn send_message(ctx: &ReducerContext, text: String) -> Result<(), String> {
	let author_id: MessageAuthorId = if let Some(author_account) = ctx.sender.try_resolve(ctx).ok()
	{
		MessageAuthorId::NativeAccountId(author_account.id)
	} else {
		MessageAuthorId::Unknown
	};

	let text = validate_message(text)?;

	log::info!("{}", text);

	ctx.db.message().insert(Message {
		id: 0,
		sender: ctx.sender,
		sent_at: ctx.timestamp,
		author_id,
		text,
	});

	Ok(())
}

#[reducer]
// Registers a message relayed from an external platform
pub fn import_message(
	ctx: &ReducerContext, author_reference: TpAccountReference, text: String,
) -> Result<(), String> {
	let author_account = author_reference.try_resolve(ctx)?;

	let sender = if let Some(owner_id) = author_account.owner_id {
		owner_id
	} else {
		ctx.sender
	};

	let text = validate_message(text)?;

	ctx.db.message().insert(Message {
		id: 0,
		sender,
		sent_at: ctx.timestamp,
		author_id: MessageAuthorId::TpAccountId(author_account.id),
		text,
	});

	Ok(())
}
