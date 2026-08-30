package com.autobrain.app

import androidx.car.app.CarContext
import androidx.car.app.Screen
import androidx.car.app.model.*

/**
 * InputScreen
 *
 * A dedicated AAOS screen that presents a [SearchTemplate] for voice or
 * keyboard text entry. Once the user submits a query, the callback fires
 * and this screen pops itself off the back-stack, returning to ChatScreen.
 *
 * AAOS Safety Note:
 *   The SearchTemplate is one of the few Car App Library templates that
 *   allows text entry. The system automatically disables the keyboard
 *   while the vehicle is moving (speed > 0 mph) to prevent distracted driving.
 *
 * @param carContext The car context provided by the AAOS host.
 * @param onQuerySubmitted Callback invoked when the user submits a query.
 */
class InputScreen(
    carContext: CarContext,
    private val onQuerySubmitted: (String) -> Unit
) : Screen(carContext) {

    private var currentQuery = ""

    override fun onGetTemplate(): Template {
        val searchCallback = object : SearchTemplate.SearchCallback {
            override fun onSearchTextChanged(searchText: String) {
                currentQuery = searchText
            }

            override fun onSearchSubmitted(searchText: String) {
                if (searchText.isNotBlank()) {
                    onQuerySubmitted(searchText)
                    screenManager.pop() // Return to ChatScreen
                }
            }
        }

        return SearchTemplate.Builder(searchCallback)
            .setHeaderAction(Action.BACK)
            .setShowKeyboardByDefault(true)
            .setInitialSearchText("")
            .setSearchHint("Ask about your Audi A3...")
            .build()
    }
}
