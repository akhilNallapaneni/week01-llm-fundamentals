package com.aicoach;

import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.anthropic.AnthropicChatModel;

import java.util.List;

public class ClinicalNoteBillingCoder {

    private static final String TRIAGE_PROMPT = """
            You are a clinical documentation specialist.
            Your job is to extract key facts from raw clinical notes for nurse triage intake.
            
            Always respond using EXACTLY this format — no deviations, no extra commentary:
            
            CHIEF_COMPLAINT: <one sentence>
            SYMPTOMS: <comma-separated list>
            DURATION: <e.g. "3 days", "onset today">
            SEVERITY: <LOW | MEDIUM | HIGH>
            FOLLOW_UP_REQUIRED: <YES | NO>
            SUMMARY: <2-3 sentence plain English summary for a non-clinical reader>
            """;

    private static final String BILLING_PROMPT = """
            You are a medical billing coder with expertise in ICD-10 and CPT coding.
            Your job is to extract billable information from clinical notes for insurance processing.
            
            Always respond using EXACTLY this format — no deviations, no extra commentary:
            
            PRIMARY_DIAGNOSIS: <primary condition being treated>
            SECONDARY_DIAGNOSES: <comma-separated list of secondary conditions>
            SUGGESTED_ICD10_CODES: <comma-separated ICD-10 codes with short labels>
            PROCEDURE_CODES: <CPT codes if any procedures mentioned, else NONE>
            BILLING_COMPLEXITY: <LOW | MEDIUM | HIGH>
            INSURANCE_NOTE: <one sentence flagging anything that may affect claim approval>
            """;

    public static void main(String[] args) {

        AnthropicChatModel model = AnthropicChatModel.builder()
                .apiKey(System.getenv("ANTHROPIC_API_KEY"))
                .modelName("claude-haiku-4-5")
                .temperature(0.0)
                .maxTokens(512)
                .build();

        // Same note — both prompts will process this
        String rawNote = """
                Pt is a 54 y/o male presenting with c/o chest tightness x3 days, \
                worsening with exertion. Reports mild SOB, no fever, no cough. \
                PMHx: HTN, DM2 managed on metformin. Current meds include lisinopril 10mg QD. \
                BP on arrival 148/92. HR 88. SpO2 97% on RA. \
                Patient denies any nausea or diaphoresis. No prior cardiac history documented. \
                Assessment pending cardiology consult.
                """;

        System.out.println("========== TRIAGE NURSE VIEW ==========");
        String triageOutput = callModel(model, TRIAGE_PROMPT, rawNote);
        System.out.println(triageOutput);

        System.out.println("\n========== BILLING CODER VIEW ==========");
        String billingOutput = callModel(model, BILLING_PROMPT, rawNote);
        System.out.println(billingOutput);

        System.out.println("\n========== KEY DIFFERENCE ==========");
        System.out.println("Same note. Same model. Same temperature.");
        System.out.println("Triage SEVERITY:     " + extractField(triageOutput, "SEVERITY"));
        System.out.println("Billing COMPLEXITY:  " + extractField(billingOutput, "BILLING_COMPLEXITY"));
    }

    private static String callModel(AnthropicChatModel model, String systemPrompt, String note) {
        var response = model.chat(
                List.of(
                        SystemMessage.from(systemPrompt),
                        UserMessage.from("Process this clinical note:\n\n" + note)
                )
        );
        return response.aiMessage().text();
    }

    private static String extractField(String output, String fieldName) {
        for (String line : output.split("\n")) {
            if (line.startsWith(fieldName + ":")) {
                return line.substring(fieldName.length() + 1).trim();
            }
        }
        return "NOT_FOUND";
    }
}