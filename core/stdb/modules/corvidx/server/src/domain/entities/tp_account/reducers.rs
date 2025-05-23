use spacetimedb::{ReducerContext, Table, reducer};

use super::{TpAccount, TpAccountReference, tp_account};
use crate::domain::entities::account_profile::{
	AccountProfile, AccountProfileMetadata, account_profile,
};

#[reducer]
/// Registers a local representation of the given 3rd party platform account.
pub fn import_tp_account(
	ctx: &ReducerContext, reference: TpAccountReference, callsign: Option<String>,
	metadata: Option<AccountProfileMetadata>,
) -> Result<(), String> {
	if ctx
		.db
		.tp_account()
		.id()
		.find(reference.to_string())
		.is_some()
	{
		return Err(format!(
			"Tp account {reference} is already registered in the system.",
		));
	}

	ctx.db.tp_account().insert(TpAccount {
		id: reference.to_string(),
		callsign,
		owner_id: None,

		profile_id: Some(
			ctx.db
				.account_profile()
				.insert(AccountProfile {
					id:       0,
					metadata: metadata.unwrap_or_default(),
				})
				.id,
		),
	});

	Ok(())
}

#[reducer]
/// Updates the local representation
/// of a 3rd party platform account handle / username.
pub fn update_tp_account_callsign(
	ctx: &ReducerContext, reference: TpAccountReference, callsign: Option<String>,
) -> Result<(), String> {
	let account = ctx
		.db
		.tp_account()
		.id()
		.find(reference.to_string())
		.ok_or(format!(
			"Tp account {reference} is not registered in the system."
		))?;

	ctx.db.tp_account().id().update(TpAccount {
		callsign,
		..account
	});

	Ok(())
}

#[reducer]
/// Updates the local representation of a 3rd party platform account profile.
pub fn update_tp_account_profile(
	ctx: &ReducerContext, reference: TpAccountReference, metadata: Option<AccountProfileMetadata>,
) -> Result<(), String> {
	let account = ctx
		.db
		.tp_account()
		.id()
		.find(reference.to_string())
		.ok_or(format!(
			"Tp account {reference} is not registered in the system."
		))?;

	let profile = if let Some(profile_id) = account.profile_id {
		ctx.db.account_profile().id().update(AccountProfile {
			id:       profile_id,
			metadata: metadata.unwrap_or_default(),
		})
	} else {
		ctx.db.account_profile().insert(AccountProfile {
			id:       0,
			metadata: metadata.unwrap_or_default(),
		})
	};

	ctx.db.tp_account().id().update(TpAccount {
		profile_id: Some(profile.id),
		..account
	});

	Ok(())
}
