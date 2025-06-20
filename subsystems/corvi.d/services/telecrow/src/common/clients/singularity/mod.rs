mod authentication;
mod handlers;

use crowdcomm_sdk::{
	configuration::corvid_subsystem::{self, ContinuitySystemConfig},
	singularity::stdb::DbConnection,
};
use spacetimedb_sdk::DbContext;

// TODO: Abstract this away to SDK,
// TODO: only passing the handlers and credential store from here
/// Loads credentials from a file and connects to the database.
pub fn connect() -> DbConnection {
	let ContinuitySystemConfig {
		module_host,
		components,
	} = corvid_subsystem::get_config();

	DbConnection::builder()
			.on_connect(handlers::on_connected)
			.on_connect_error(handlers::on_connect_error)
			.on_disconnect(handlers::on_disconnected)
			// If the account has previously connected, we'll have saved a token in the `on_connect` callback.
			// In that case, we'll load it and pass it to `with_token`,
			// so we can re-authenticate as the same `Identity`.
			.with_token(authentication::credential_store().load()
				.expect("Error loading credentials")
			)
			.with_module_name(components.singularity.module_name)
			.with_uri(module_host)
			.build()
			.expect("Failed to connect")
}

/// Registers subscriptions to tables.
pub fn subscribe_to_tables(ctx: &DbConnection) {
	ctx
		.subscription_builder()
		.on_applied(handlers::on_sub_applied)
		.on_error(handlers::on_sub_error)
		// Facilitating creation of a local partial replica of the database.
		.subscribe([
			"SELECT * FROM external_authentication_request",
			"SELECT * FROM actor_profile",
			"SELECT * FROM external_actor",
			"SELECT * FROM message",
			// "SELECT * FROM message_channel",
			"SELECT * FROM account",
		]);
}
